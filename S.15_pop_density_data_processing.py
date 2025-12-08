import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;

matplotlib.use('Qt5Agg')
import scipy.stats as st
import warnings

warnings.filterwarnings("ignore")
import os
import tifffile as tf
import cv2
from joblib import Parallel, delayed


def extend_edge(array):
    array1 = array * 1
    array1[1:, :] = array[:-1, :]  # Shift elements up
    array2 = array * 1
    array2[:-1, :] = array[1:, :]  # Shift elements down
    array3 = array * 1
    array3[:, 1:] = array[:, :-1]  # Shift elements left
    array4 = array * 1
    array4[:, :-1] = array[:, 1:]  # Shift elements right
    array5 = array * 1
    array5[:-1, 1:] = array[1:, :-1]  # Shift elements up-left
    array6 = array * 1
    array6[:-1, :-1] = array[1:, 1:]  # Shift elements up-right
    array7 = array * 1
    array7[1:, 1:] = array[:-1, :-1]  # Shift elements down-right
    array8 = array * 1
    array8[1:, :-1] = array[:-1, 1:]  # Shift elements down-left
    stacked_array = np.stack([array1, array2, array3, array4, array5, array6, array7, array8], axis=0)
    result = np.nanmean(stacked_array, axis=0)
    result[result > 0] = 1
    return result


def process_id(id):
    try:
        lst = tf.imread(lst_dir_path + '/Landsat_' + str(id) + '.tif')
    except FileNotFoundError:
        return None
    lst = np.nanmean(lst, axis=2)
    treeC = tf.imread(treeC_folder + '/cropC_' + str(id) + '.tif')
    grassC = tf.imread(grassC_folder + '/grassC_' + str(id) + '.tif')
    crop_frc = tf.imread(crop_folder + '/cropC_' + str(id) + '.tif')
    crop_frc = cv2.resize(crop_frc, (lst.shape[1], lst.shape[0]), cv2.INTER_LINEAR)

    ndvi = tf.imread(ndvi_folder + '/Landsat_ndvi_' + str(id) + '.tif')
    vegC = treeC + grassC
    crop_frc[crop_frc > 0.2] = np.nan

    urbanExp_frc = tf.imread(urbanExp_folder + '/urban_exp_C_' + str(id) + '.tif')
    urbanExp_frc = cv2.resize(urbanExp_frc, (lst.shape[1], lst.shape[0]), cv2.INTER_LINEAR)

    pop = tf.imread(pop_folder + '/pop_' + str(id) + '.tif')
    pop = cv2.resize(pop, (lst.shape[1], lst.shape[0]), cv2.INTER_LINEAR)
    mask = np.isnan(urbanExp_frc>0.2) | np.isnan(crop_frc)#np.isnan(dem)
    ndvi[mask] = np.nan
    pop[mask] = np.nan
    vegC[mask] = np.nan

    urban_1990 = tf.imread(urban_folder_1990 + '/urban_1990_' + str(id) + '.tif').astype(float)
    urban_1990 = cv2.resize(urban_1990, (lst.shape[1], lst.shape[0]), cv2.INTER_NEAREST)
    urban_1990[urban_1990 == 0] = np.nan

    urban_2018 = tf.imread(urban_folder_2018 + '/fvc_' + str(id) + '.tif').astype(float)
    urban_2018_rsz = cv2.resize(urban_2018, (lst.shape[1], lst.shape[0]), cv2.INTER_NEAREST)
    urban_2018_rsz[urban_2018_rsz != float(id)] = np.nan
    rural_near = urban_2018_rsz * 1  # rural-urban interface
    for i in range(3 * 10):  # 3km
        rural_near = extend_edge(rural_near)

    rural_bgr = rural_near * 1  # rural background
    for i in range(10 * 10):  # 10km
        rural_bgr = extend_edge(rural_bgr)
    rural_bgr[~np.isnan(rural_near)] = np.nan

    # dem = tf.imread(dem_folder + '/dem_' + str(id) + '.tif').astype(float)
    # dem_core = np.nanmean(dem[~np.isnan(urban_1990)])
    # dem[(dem > dem_core + 50) | (dem < dem_core - 50)] = np.nan
    # mask = np.isnan(urbanExp_frc > 0.2) | np.isnan(dem)
    # vegC[mask] = np.nan

    pop_core = np.nanmean(pop[~np.isnan(urban_1990)])
    pop_bg = np.nanmean(pop[~np.isnan(rural_bgr)])
    print(id)

    return [id, pop_core, pop_bg]


## main ##
lst_dir_path = os.path.join('..', '..', 'project 1 urban vegetation thermoregulation', '1_Input', 'LST')

treeC_folder = os.path.join('..', '..', 'project 1 urban vegetation thermoregulation', '1_Input', 'treeCover_100m')

grassC_folder = os.path.join('..', '..', 'project 1 urban vegetation thermoregulation', '1_Input', 'grassCover_100m')
crop_folder = os.path.join('..', '..', 'project 1 urban vegetation thermoregulation', '1_Input', 'cropCover_100m')

urbanExp_folder = os.path.join('..', '1_Input', 'urban_expansion_frac_500m')

ta_ERA_folder = os.path.join('..', '1_Input', 'ta_anom')
ta_filenames = os.listdir(ta_ERA_folder)
ndvi_folder = os.path.join('..', '..', 'project 1 urban vegetation thermoregulation', '1_Input', 'NDVI')
urban_folder_1990 = os.path.join('..', '1_Input', 'urban', 'urban_1990_250m')
urban_folder_2018 = os.path.join('..', '1_Input', 'urban', 'urban_2018_1000m')
dem_folder = os.path.join('..', '..', 'project 1 urban vegetation thermoregulation', '1_Input', 'DEM_100m')
pop_folder = os.path.join('..', '..', 'urban_env_data', 'Population_density')

IDs = []
for name in ta_filenames:
    id = float(name.split('_')[-1].split('.npy')[0])
    IDs.append(id)
IDs = np.sort(IDs)

# Use parallel processing to speed up the loop
results = Parallel(n_jobs=-1)(delayed(process_id)(id) for id in IDs)

# Filter out None results
results = [result for result in results if result is not None]

results_arr = np.array(results)
result_filtered = []
for i in range(len(results)):
    rst = results[i] * 1
    result_filtered.append(rst)
results_arr = np.array(result_filtered)
# Convert results_arr to DataFrame
columns = ['ID', 'pop_core_all', 'pop_bg_all']

results_df = pd.DataFrame(results_arr, columns=columns)

# Print the DataFrame
print(results_df)
output_dir = os.path.join('..', '2_Output', 'drivers', 'pop.csv')
results_df.to_csv(output_dir, index=False)