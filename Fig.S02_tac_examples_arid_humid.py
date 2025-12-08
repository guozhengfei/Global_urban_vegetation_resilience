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
    TACs = np.load(current_dir + '/2_Output/tac_nadir_city_3zones.npz')['array1']
    ID = np.load(current_dir + '/2_Output/tac_nadir_city_3zones.npz')['array2']
    TACs_mean = np.nanmean(TACs, axis=2)
    tac_global_mean = np.nanmean(TACs_mean, axis=0)
    df_tac = pd.DataFrame(TACs_mean)
    df_tac.columns = ['urban_core', 'urban_edge', 'rural_bgr']
    df_tac['ID'] = ID
    df_tac['tac_diff'] = df_tac['urban_core'] - df_tac['rural_bgr']

    # Initialize a list to store valid_num values
    valid_nums = []

    # Paths to data folders
    tac_folder = current_dir + '/2_Output/tac_500m_yr/'
    urban_folder_2018 = current_dir + '/1_Input/urban/urban_2018_250m/'
    urban_folder_1990 = current_dir + '/1_Input/urban/urban_1990_250m/'

    # Calculate valid_num for each ID
    for id in ID:
        urban2018 = tf.imread(urban_folder_2018 + 'urban_' + str(id) + '.tif').astype(float)
        valid_num = np.sum(~np.isnan(urban2018))  # Count non-NaN values
        valid_nums.append(valid_num)  # Append to the list

    # Add valid_num to the DataFrame
    df_tac['valid_num'] = valid_nums

    # Sort the DataFrame by tac_diff
    df_tac = df_tac.sort_values(by='tac_diff', ascending=True, ignore_index=True)
    ids = ['767.0', '1096.0', '844.0', '60.0']  # Get four IDs for the subplots

    # Create a figure with four subplots
    fig, axes = plt.subplots(1, 4, figsize=(11.3*0.7, 3.8*0.7))  # 1x4 grid of subplots

    # Desired aspect ratio
    target_aspect_ratio = 1.05 /1

    for i, ax in enumerate(axes.flat):  # Iterate over each subplot
        id = ids[i]  # Get the current ID

        # Load TAC data
        tac_i = np.load(tac_folder + 'tac_' + id + '.npy')[:, :, 2:-2]
        tac_spa_mean = np.nanmean(tac_i, axis=2)

        # Calculate target dimensions for resizing
        original_height, original_width = tac_spa_mean.shape
        target_width = original_width
        target_height = int(target_width * target_aspect_ratio)

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
        rural_bgr = extend_edge_1(rural_bgr, 20)
        rural_bgr = extend_edge(extend_edge(extend_edge(extend_edge(extend_edge(rural_bgr)))))
        rural_bgr[~np.isnan(rural_near)] = 1
        rural_bgr_edge = extend_edge(rural_bgr)
        rural_bgr_edge[~np.isnan(rural_bgr)] = np.nan

        # Mask TAC data
        tac_spa_mean_resized[np.isnan(rural_bgr)] = np.nan
        p05 = np.nanpercentile(tac_spa_mean_resized, 9)
        p95 = np.nanpercentile(tac_spa_mean_resized, 95)

        # Plot TAC spatial mean
        cmap = plt.cm.RdBu_r
        cmap.set_bad(color='none')  # Set NaN regions to light grey
        extent = [min_lon, max_lon, min_lat, max_lat]  # Geographic extent
        im = ax.imshow(np.ma.masked_invalid(tac_spa_mean_resized), cmap=cmap, vmin=p05, vmax=p95, alpha=0.65)

        # Create a combined mask for edges
        combined_edges = np.zeros_like(urban_1990_edge) + np.nan
        combined_edges[~np.isnan(urban_1990_edge)] = 1  # Urban 1990 edge
        combined_edges[~np.isnan(rural_near_edge)] = 2  # Rural-near edge
        combined_edges[~np.isnan(rural_bgr_edge)] = 3  # Rural background edge

        # Define a custom colormap and normalization
        custom_cmap = ListedColormap(['#1b9e77', '#7570b3', '#d95f02'])  # Colors for urban, rural-near, rural-bgr
        custom_norm = BoundaryNorm([0, 1, 2, 3], custom_cmap.N)
        ax.imshow(combined_edges, cmap=custom_cmap)

        # Set axis labels and ticks
        ax.set_xlabel('Longitude')

        # Generate tick positions and labels
        ytick_positions = np.linspace(0, tac_spa_mean_resized.shape[0], 4).astype(int)  # Y-axis positions
        ytick_labels = np.linspace(min_lat, max_lat, 4)  # Y-axis labels (latitude)
        xtick_positions = np.linspace(0, tac_spa_mean_resized.shape[1], 4).astype(int)  # X-axis positions
        xtick_labels = np.linspace(min_lon, max_lon, 4)  # X-axis labels (longitude)

        # Set ticks and format labels to two decimal places
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels([f"{label:.1f}" for label in ytick_labels])  # Format latitude labels
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([f"{label:.1f}" for label in xtick_labels])  # Format longitude labels

    # Save the figure
    figToPath = current_dir + '/4_Figures/FigS02_tac_arid_humid'

    # Adjust subplot spacing
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.32)  # Reduce horizontal spacing (default is 0.2, decrease this value for less space)

    # Save and show the figure
    plt.savefig(figToPath, dpi=900)
    plt.show()