import numpy as np
import tifffile as tf
import matplotlib;
# matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
plt.close()

import os
import warnings
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
warnings.filterwarnings("ignore")
import cv2
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
def extend_edge_1(array, distance):
    array1 = array*1
    array1[np.isnan(array1)] = 0
    array1 = array1[distance:-distance, distance:-distance]
    array1 = cv2.resize(array1, (array.shape[1], array.shape[0]), interpolation=cv2.INTER_NEAREST)
    array1[array1 > 0] = 1
    array1[array1<=0] = np.nan
    return array1

if __name__ == '__main__':
    # Identify the urban_rural tac diff, and sort it
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

    # Paths to data folders
    filename = current_dir + '/1_Input/nadir_ndvi_monthly_500/nadir_ndvi_monthly_6.0.tif'
    urban_folder_2018 = current_dir + '/1_Input/urban/urban_2018_250m/'
    urban_folder_1990 = current_dir + '/1_Input/urban/urban_1990_250m/'


    # Read data using rasterio
    with rasterio.open(filename) as src:
        ndvi = src.read()  # Read all bands
        transform = src.transform

    # Calculate mean NDVI across time dimension
    ndvi_mean = np.nanmean(ndvi, axis=0)
    id = '6.0'  # Get four IDs for the subplots

    # Create a figure with four subplots
    fig, ax = plt.subplots(1, figsize=(5.2, 3.5))  # 1x4 grid of subplots

    # Desired aspect ratio
    target_aspect_ratio = 1/1

    # Load TAC data
    tac_spa_mean = ndvi_mean

    # Calculate target dimensions for resizing
    original_height, original_width = tac_spa_mean.shape
    target_width = original_width
    target_height = original_height# int(target_width * target_aspect_ratio)

    # Resize the array to match the target aspect ratio
    tac_spa_mean_resized = cv2.resize(tac_spa_mean, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # Load and process urban data
    urban_2018_path = urban_folder_2018 + 'urban_' + id + '.tif'
    with rasterio.open(urban_2018_path) as src:
        bounds = src.bounds  # Get geographic bounds
        min_lon, max_lon = bounds.left, bounds.right
        min_lat, max_lat = bounds.bottom, bounds.top

    urban_2018 = tf.imread(urban_2018_path).astype(float)
    urban_2018_rsz = cv2.resize(urban_2018, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    urban_2018_rsz[urban_2018_rsz != float(id)] = np.nan
    urban_1990 = tf.imread(urban_folder_1990 + 'urban_1990_' + id + '.tif').astype(float)
    urban_1990 = cv2.resize(urban_1990, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    urban_1990[urban_1990 == 0] = np.nan
    urban_1990[~np.isnan(urban_1990)] = 1

    urban_1990_edge = extend_edge(extend_edge(urban_1990))
    urban_1990_edge[~np.isnan(urban_1990)] = np.nan

    rural_near = urban_2018_rsz * 1  # Rural-urban interface
    rural_near = extend_edge(extend_edge(rural_near))
    rural_near_edge = extend_edge(extend_edge(rural_near))
    rural_near_edge[~np.isnan(rural_near)] = np.nan

    rural_bgr = rural_near * 1  # Rural background
    rural_bgr = extend_edge_1(rural_bgr, 30)
    rural_bgr = extend_edge(extend_edge(extend_edge(extend_edge(extend_edge(rural_bgr)))))
    rural_bgr[~np.isnan(rural_near)] = 1
    rural_bgr_edge = extend_edge(rural_bgr)
    rural_bgr_edge[~np.isnan(rural_bgr)] = np.nan

    # Mask TAC data
    tac_spa_mean_resized[np.isnan(rural_bgr)] = np.nan
    p05 = np.nanpercentile(tac_spa_mean_resized, 9)
    p95 = np.nanpercentile(tac_spa_mean_resized, 95)

    # Plot TAC spatial mean
    cmap = plt.cm.Greens
    cmap.set_bad(color='none')  # Set NaN regions to light grey
    extent = [min_lon, max_lon, min_lat, max_lat]  # Geographic extent
    im = ax.imshow(np.ma.masked_invalid(tac_spa_mean_resized), cmap=cmap, vmin=p05, vmax=p95, alpha=0.65)

    # Add colorbar for NDVI values
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('NDVI', fontsize=12)

    # Create a combined mask for edges
    combined_edges = np.zeros_like(urban_1990_edge) + np.nan
    combined_edges[~np.isnan(urban_1990_edge)] = 1  # Urban 1990 edge
    combined_edges[~np.isnan(rural_near_edge)] = 2  # Rural-near edge
    combined_edges[~np.isnan(rural_bgr_edge)] = 3  # Rural background edge

    # Define a custom colormap and normalization
    custom_cmap = ListedColormap(['#1b9e77', '#7570b3', '#d95f02'])  # Colors for urban, rural-near, rural-bgr
    custom_norm = BoundaryNorm([0, 1, 2, 3], custom_cmap.N)
    edges_im = ax.imshow(combined_edges, cmap=custom_cmap)

    # Add legend for edges
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1b9e77', label='Urban 1990'),
        Patch(facecolor='#7570b3', label='Rural-near'),
        Patch(facecolor='#d95f02', label='Rural background')
    ]

    # Set axis labels and ticks
    ax.set_xlabel('Longitude')

    # Generate tick positions and labels
    ytick_positions = np.linspace(0, tac_spa_mean_resized.shape[0], 4).astype(int)  # Y-axis positions
    ytick_labels = np.linspace(min_lat, max_lat, 4)  # Y-axis labels (latitude)
    xtick_positions = np.linspace(0, tac_spa_mean_resized.shape[1], 4).astype(int)  # X-axis positions
    xtick_labels = np.linspace(min_lon, max_lon, 4)  # X-axis labels (longitude)

    # Set ticks and format labels to two decimal places
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([f"{label:.2f}" for label in ytick_labels])  # Format latitude labels
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f"{label:.2f}" for label in xtick_labels])  # Format longitude labels

    figToPath = current_dir + '/4_Figures/FigS05g_ndvi_example'
    plt.tight_layout()
    plt.savefig(figToPath, dpi=900)

    # remove long-term mean
    yr_num = 22
    bands_year = 12
    ser = tf.imread(filename)[:,:,:-2]
    ser = np.reshape(ser,[ser.shape[0]*ser.shape[1],ser.shape[2]])
    EVI_yr = np.zeros_like(ser)
    for year in range(yr_num):
        st = year * bands_year
        ed = st + bands_year
        evi_year = np.nanmean(ser[:, st:ed], axis=1)
        evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
        EVI_yr[:, st:ed] = evi_year_rep
    rm_offline = ser - EVI_yr
    # plt.figure(); plt.plot(np.nanmean(rm_offline, axis=0))

    # remove seasonality
    Evi_sea_rep = np.zeros_like(rm_offline)
    for yr in range(yr_num):
        start_index = (yr - 3) * bands_year
        if start_index < 0: start_index = 0
        end_index = (yr + 4) * bands_year
        if end_index > bands_year * yr_num: end_index = bands_year * yr_num
        data_i = rm_offline[:, start_index:end_index]
        Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)),
                          axis=1)
        Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea

    res = rm_offline - Evi_sea_rep
    res[np.isnan(res)] = 0

    res_uc = res[~np.isnan(urban_1990).flatten(),:]
    res_rb = res[~np.isnan(rural_bgr_edge).flatten(),:]

    fig, ax = plt.subplots(1, figsize=(8.2, 3.5))  # 1x4 grid of subplots
    plt.plot(np.nanmean(res_uc,axis=0))
    plt.plot(np.nanmean(res_rb, axis=0),alpha=0.6)
    ax.set_ylim([-0.10,0.06])
    ax.set_xticks([0,60,120,180,240],['2000','2005','2010','2015','2020'])
    ax.set_xlabel('Year')
    ax.set_ylabel('Redidual kNDVI')
    figToPath = current_dir + '/4_Figures/FigS05g_urban_rural_ndvi_variability'
    plt.tight_layout()
    plt.savefig(figToPath, dpi=600)


