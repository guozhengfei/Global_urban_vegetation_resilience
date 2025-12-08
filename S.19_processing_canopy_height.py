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
    CH = tf.imread(ch_folder + '/canopyHeight_' + str(id) + '.tif')

    urban_1990 = tf.imread(urban_folder_1990 + '/urban_1990_' + str(id) + '.tif').astype(float)
    urban_1990 = cv2.resize(urban_1990, (CH.shape[1], CH.shape[0]), cv2.INTER_NEAREST)
    urban_1990[urban_1990 == 0] = np.nan

    urban_2018 = tf.imread(urban_folder_2018 + '/fvc_' + str(id) + '.tif').astype(float)
    urban_2018_rsz = cv2.resize(urban_2018, (CH.shape[1], CH.shape[0]), cv2.INTER_NEAREST)
    urban_2018_rsz[urban_2018_rsz != float(id)] = np.nan
    rural_near = urban_2018_rsz * 1  # rural-urban interface
    for i in range(10):  # 3km
        rural_near = extend_edge(rural_near)

    rural_bgr = rural_near * 1  # rural background
    for i in range(4 * 10):  # 10km
        rural_bgr = extend_edge(rural_bgr)
    rural_bgr[~np.isnan(rural_near)] = np.nan



    ch_core = np.nanmean(CH[~np.isnan(urban_1990)])
    ch_bg = np.nanmean(CH[~np.isnan(rural_bgr)])
    print(id)

    return [id, ch_core, ch_bg]


## main ##
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
urban_folder_1990 = os.path.join('..', '1_Input', 'urban', 'urban_1990_250m')
urban_folder_2018 = os.path.join('..', '1_Input', 'urban', 'urban_2018_1000m')
ch_folder = os.path.join('..', '..', 'urban_env_data', 'canopy_height')
filenames = os.listdir(ch_folder)

IDs = []
for name in filenames:
    id = float(name.split('_')[-1].split('.tif')[0])
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
columns = ['ID', 'ch_core', 'ch_bg']

results_df = pd.DataFrame(results_arr, columns=columns)

# Print the DataFrame
print(results_df)
output_dir = os.path.join('..', '2_Output', 'drivers', 'canopy_height.csv')
results_df.to_csv(output_dir, index=False)