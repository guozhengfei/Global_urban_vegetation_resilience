# extract pure veg pixels in urban and rural region
import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st
import warnings
import os
import tifffile as tf
import cv2
from joblib import Parallel, delayed
import rasterio

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
        with rasterio.open(lst_dir_path + '/Landsat_' + str(id) + '.tif') as src:
            lst = src.read()
            lst_transform = src.transform
            lst = np.nanmean(lst, axis=0)
        with rasterio.open(treeC_folder + '/cropC_' + str(id) + '.tif') as src:
            treeC = src.read(1)
        with rasterio.open(grassC_folder + '/grassC_' + str(id) + '.tif') as src:
            grassC = src.read(1)
        vegC = treeC + grassC
        with rasterio.open(urbanExp_folder + '/urban_exp_C_' + str(id) + '.tif') as src:
            urbanExp_frc = src.read(1)
        urbanExp_frc = cv2.resize(urbanExp_frc, (lst.shape[1], lst.shape[0]), cv2.INTER_LINEAR)

        with rasterio.open(urban_folder_1990 + '/urban_1990_' + str(id) + '.tif') as src:
            urban_1990 = src.read(1).astype(float)
        urban_1990 = cv2.resize(urban_1990, (lst.shape[1], lst.shape[0]), cv2.INTER_NEAREST)
        urban_1990[urban_1990 == 0] = np.nan

        with rasterio.open(urban_folder_2018 + '/fvc_' + str(id) + '.tif') as src:
            urban_2018 = src.read(1).astype(float)
        urban_2018_rsz = cv2.resize(urban_2018, (lst.shape[1], lst.shape[0]), cv2.INTER_NEAREST)
        urban_2018_rsz[urban_2018_rsz != float(id)] = np.nan
        rural_near = urban_2018_rsz * 1
        for i in range(3*10):
            rural_near = extend_edge(rural_near)

        rural_bgr = rural_near * 1
        for i in range(10*10):
            rural_bgr = extend_edge(rural_bgr)

        with rasterio.open(dem_folder + '/dem_' + str(id) + '.tif') as src:
            dem = src.read(1).astype(float)
        dem_core = np.nanmean(dem[~np.isnan(urban_1990)])
        dem[(dem > dem_core + 50) | (dem < dem_core - 50)] = np.nan
        mask = np.isnan(urbanExp_frc > 0.2) | np.isnan(dem)
        vegC[mask] = np.nan

        vegC_core = vegC * 1
        vegC_core[np.isnan(urban_1990)] = np.nan
        vegC_core_flat = vegC_core.flatten()
        valid_indices_core = np.where(~np.isnan(vegC_core_flat))[0]
        # top_100_indices_core = valid_indices_core[np.argpartition(vegC_core_flat[valid_indices_core], -100)[-100:]]
        top_100_indices_core = valid_indices_core[np.argpartition(vegC_core_flat[valid_indices_core], -100)[-100:]]
        fvc_core = vegC_core_flat[top_100_indices_core]
        treeC_core = treeC.flatten()[top_100_indices_core]
        grassC_core = grassC.flatten()[top_100_indices_core]
        rows_core, cols_core = np.unravel_index(top_100_indices_core, vegC_core.shape)

        rural_bgr[~np.isnan(rural_near)] = np.nan
        vegC_bg = vegC * 1
        vegC_bg[np.isnan(rural_bgr)] = np.nan
        vegC_bg_flat = vegC_bg.flatten()
        valid_indices_bg = np.where(~np.isnan(vegC_bg_flat))[0]
        top_100_indices_bg = valid_indices_bg[np.argpartition(vegC_bg_flat[valid_indices_bg], -100)[-100:]]
        fvc_bg = vegC_bg_flat[top_100_indices_bg]
        treeC_bg = treeC.flatten()[top_100_indices_bg]
        grassC_bg = grassC.flatten()[top_100_indices_bg]
        rows_bg, cols_bg = np.unravel_index(top_100_indices_bg, vegC_bg.shape)

        # Use rasterio transform to get coordinates
        core_coords = [rasterio.transform.xy(lst_transform, r, c, offset='center') for r, c in zip(rows_core, cols_core)]
        bg_coords = [rasterio.transform.xy(lst_transform, r, c, offset='center') for r, c in zip(rows_bg, cols_bg)]

        core_lat = [coord[1] for coord in core_coords]
        core_lon = [coord[0] for coord in core_coords]
        bg_lat = [coord[1] for coord in bg_coords]
        bg_lon = [coord[0] for coord in bg_coords]

        output = {
            'id': id,
            'core_lat': core_lat,
            'core_lon': core_lon,
            'core_fvc': fvc_core.tolist(),
            'core_tree_frac': treeC_core.tolist(),
            'core_grass_frac': grassC_core.tolist(),
            'bg_lat': bg_lat,
            'bg_lon': bg_lon,
            'bg_fvc': fvc_bg.tolist(),
            'bg_tree_frac': treeC_bg.tolist(),
            'bg_grass_frac': grassC_bg.tolist()
        }
        return output
    except FileNotFoundError:
        return None

## main ##
lst_dir_path = os.path.join('/Volumes/Zhengfei_02/Zhengfei_01-BACKUP', 'project 1 urban vegetation thermoregulation', '1_Input', 'LST')

treeC_folder = os.path.join('/Volumes/Zhengfei_02/Zhengfei_01-BACKUP', 'project 1 urban vegetation thermoregulation', '1_Input', 'treeCover_100m')

grassC_folder = os.path.join('/Volumes/Zhengfei_02/Zhengfei_01-BACKUP', 'project 1 urban vegetation thermoregulation', '1_Input', 'grassCover_100m')

urbanExp_folder = os.path.join( '..', '1_Input', 'urban_expansion_frac_500m')
urban_folder_1990 = os.path.join('..','1_Input','urban','urban_1990_250m')
urban_folder_2018 = os.path.join('..','1_Input','urban','urban_2018_1000m')

dem_folder = os.path.join('/Volumes/Zhengfei_02/Zhengfei_01-BACKUP','project 1 urban vegetation thermoregulation','1_Input','DEM_100m')

filenames = os.listdir(lst_dir_path)
IDs = []
for name in filenames:
    id = float(name.split('_')[-1].split('.tif')[0])
    IDs.append(id)
IDs = np.sort(IDs)[1:]

# Use parallel processing to speed up the loop
results = Parallel(n_jobs=-1)(delayed(process_id)(id) for id in IDs)
results = [r for r in results if r is not None]  # Remove None results

# Convert results to DataFrame and explode lists
df = pd.DataFrame(results)

# Explode core and bg columns separately, then concatenate
core_df = df[['id', 'core_lat', 'core_lon', 'core_fvc', 'core_tree_frac', 'core_grass_frac']].explode(
    ['core_lat', 'core_lon', 'core_fvc', 'core_tree_frac', 'core_grass_frac']
)
core_df['type'] = 'core'
core_df = core_df.rename(columns={
    'core_lat': 'lat',
    'core_lon': 'lon',
    'core_fvc': 'fvc',
    'core_tree_frac': 'tree_frac',
    'core_grass_frac': 'grass_frac'
})

bg_df = df[['id', 'bg_lat', 'bg_lon', 'bg_fvc', 'bg_tree_frac', 'bg_grass_frac']].explode(
    ['bg_lat', 'bg_lon', 'bg_fvc', 'bg_tree_frac', 'bg_grass_frac']
)
bg_df['type'] = 'bg'
bg_df = bg_df.rename(columns={
    'bg_lat': 'lat',
    'bg_lon': 'lon',
    'bg_fvc': 'fvc',
    'bg_tree_frac': 'tree_frac',
    'bg_grass_frac': 'grass_frac'
})

final_df = pd.concat([core_df, bg_df], ignore_index=True)

# Save to CSV
output_dir = os.path.join('..', '2_Output', 'pure_veg', 'urban_factors_effect_100.csv')
final_df.to_csv(output_dir, index=False)
