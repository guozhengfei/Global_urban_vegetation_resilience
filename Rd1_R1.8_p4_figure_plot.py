import numpy as np
import tifffile as tf
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import os
import warnings

plt.rc('font', family='Arial')

warnings.filterwarnings("ignore")
import cv2
import rasterio

TITLE_SIZE = 13+2
AXIS_LABEL_SIZE = 12+2
TICK_LABEL_SIZE = 11+2
LEGEND_SIZE = 12+2


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
    array1 = array * 1
    array1[np.isnan(array1)] = 0
    array1 = array1[distance:-distance, distance:-distance]
    array1 = cv2.resize(array1, (array.shape[1], array.shape[0]), interpolation=cv2.INTER_NEAREST)
    array1[array1 > 0] = 1
    array1[array1 <= 0] = np.nan
    return array1


def stable_landcover_mask(landcover_file, target_shape):
    landcover = tf.imread(landcover_file)
    if landcover.ndim != 3:
        raise ValueError(f'Expected a 3-band land cover tif: {landcover_file}')
    if landcover.shape[0] == 3 and landcover.shape[-1] != 3:
        landcover = np.moveaxis(landcover, 0, -1)
    if landcover.shape[-1] != 3:
        raise ValueError(f'Expected exactly 3 land cover bands: {landcover_file}')

    stable = landcover[:, :, 1] == landcover[:, :, 2]
    if stable.shape != target_shape:
        stable = cv2.resize(stable.astype(np.uint8), (target_shape[1], target_shape[0]), cv2.INTER_NEAREST).astype(bool)

    mask = stable.astype(float)
    mask[~stable] = np.nan
    return mask


def resize_nearest(array, target_shape):
    if array.shape == target_shape:
        return array
    return cv2.resize(array, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


def resize_linear(array, target_shape):
    if array.shape == target_shape:
        return array
    return cv2.resize(array, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def build_fig01g_boundaries(urban_2018, urban_1990, id, target_shape):
    urban_2018_rsz = resize_nearest(urban_2018, target_shape).astype(float)
    urban_2018_rsz[urban_2018_rsz != float(id)] = np.nan

    urban_1990 = resize_nearest(urban_1990, target_shape).astype(float)
    urban_1990[urban_1990 == 0] = np.nan
    urban_1990[~np.isnan(urban_1990)] = 1

    urban_1990_edge = extend_edge(extend_edge(urban_1990))
    urban_1990_edge[~np.isnan(urban_1990)] = np.nan

    rural_near = urban_2018_rsz * 1
    rural_near = extend_edge(extend_edge(rural_near))
    rural_near_edge = extend_edge(extend_edge(rural_near))
    rural_near_edge[~np.isnan(rural_near)] = np.nan

    rural_bgr = rural_near * 1
    rural_bgr = extend_edge_1(rural_bgr, 20)
    rural_bgr = extend_edge(extend_edge(extend_edge(extend_edge(extend_edge(rural_bgr)))))
    rural_bgr[~np.isnan(rural_near)] = 1
    rural_bgr_edge = extend_edge(rural_bgr)
    rural_bgr_edge[~np.isnan(rural_bgr)] = np.nan

    combined_edges = np.zeros_like(urban_1990_edge) + np.nan
    combined_edges[~np.isnan(urban_1990_edge)] = 1
    combined_edges[~np.isnan(rural_near_edge)] = 2
    combined_edges[~np.isnan(rural_bgr_edge)] = 3
    return combined_edges, rural_bgr


def mask_outside_rural_bgr(data, rural_bgr):
    clipped = data.astype(float, copy=True)
    clipped[np.isnan(rural_bgr)] = np.nan
    return clipped


def set_geo_ticks(ax, data_shape, bounds):
    min_lon, max_lon, min_lat, max_lat = bounds
    ytick_positions = np.linspace(0, data_shape[0], 4).astype(int)
    xtick_positions = np.linspace(0, data_shape[1], 4).astype(int)
    ytick_labels = np.linspace(min_lat, max_lat, 4)
    xtick_labels = np.linspace(min_lon, max_lon, 4)

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([f'{label:.1f}' for label in ytick_labels], fontsize=TICK_LABEL_SIZE)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f'{label:.1f}' for label in xtick_labels], fontsize=TICK_LABEL_SIZE)
    ax.tick_params(width=0.8, length=3)


def plot_panel(ax, data, title, boundary_edges, bounds, cmap='viridis', vmin=None, vmax=None):
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.75)
    edge_cmap = ListedColormap(['#810f7c', '#000000', '#e31a1c'])
    ax.imshow(boundary_edges, cmap=edge_cmap)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel('Longitude', fontsize=AXIS_LABEL_SIZE)
    set_geo_ticks(ax, data.shape, bounds)
    return im

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/nadir_ndvi_15d_500/'
    grass_folder = current_dir + '/1_Input/grassCover_250m/'
    tree_folder = current_dir + '/1_Input/treeCover_250m/'
    crop_folder = current_dir + '/1_Input/cropCover_250m/'
    urbanExp_folder = current_dir + '/1_Input/urban_expansion_frac_500m/'
    urban_folder_2018 = current_dir + '/1_Input/urban/urban_2018_250m/'
    urban_folder_1990 = current_dir + '/1_Input/urban/urban_1990_250m/'
    landcover_folder = current_dir + '/1_Input/landcover_modis_500m/'

    id = 334.0# using Beijing as an example
    EVI_file = EVI_folder + 'nadir_ndvi_15d_' + str(id) + '.tif'
    EVI0 = tf.imread(EVI_file)[:, :, :264 * 2][:, :, ::2]
    crop_frc = tf.imread(crop_folder + 'cropC_' + str(id) + '.tif')
    crop_frc = cv2.resize(crop_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)

    grass_frc = tf.imread(grass_folder + 'grassC_' + str(id) + '.tif')
    grass_frc = cv2.resize(grass_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)

    tree_frc = tf.imread(tree_folder + 'treeC_' + str(id) + '.tif')
    tree_frc = cv2.resize(tree_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)

    urbanExp_frc = tf.imread(urbanExp_folder + 'urban_exp_C_' + str(id) + '.tif')
    if urbanExp_frc.shape != EVI0.shape[:2]:
        urbanExp_frc = cv2.resize(urbanExp_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
    landcover_stable = stable_landcover_mask(landcover_folder + 'landcover_500m_' + str(id) + '.tif', EVI0.shape[:2])

    mean_ndvi = np.nanmean(EVI0, axis=2)
    veg_nature = grass_frc + tree_frc
    crop_mask = (crop_frc > 0.2) | (crop_frc > veg_nature)
    urban_expansion_mask = urbanExp_frc > 0.2
    landcover_change_mask = np.isnan(landcover_stable)
    mask = crop_mask | urban_expansion_mask | landcover_change_mask
    remain_ndvi = mean_ndvi.copy()
    remain_ndvi[mask] = np.nan

    urban_2018_path = urban_folder_2018 + 'urban_' + str(id) + '.tif'
    with rasterio.open(urban_2018_path) as src:
        bounds = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)

    urban_2018 = tf.imread(urban_2018_path).astype(float)
    urban_1990 = tf.imread(urban_folder_1990 + 'urban_1990_' + str(id) + '.tif').astype(float)

    target_aspect_ratio = 1.05 / 1
    target_width = EVI0.shape[1]
    target_height = int(target_width * target_aspect_ratio)
    target_shape = (target_height, target_width)

    mean_ndvi = resize_linear(mean_ndvi, target_shape)
    remain_ndvi = resize_linear(remain_ndvi, target_shape)
    crop_mask = resize_nearest(crop_mask.astype(np.uint8), target_shape).astype(bool)
    urban_expansion_mask = resize_nearest(urban_expansion_mask.astype(np.uint8), target_shape).astype(bool)
    landcover_change_mask = resize_nearest(landcover_change_mask.astype(np.uint8), target_shape).astype(bool)

    boundary_edges, rural_bgr = build_fig01g_boundaries(urban_2018, urban_1990, id, target_shape)
    mean_ndvi = mask_outside_rural_bgr(mean_ndvi, rural_bgr)
    remain_ndvi = mask_outside_rural_bgr(remain_ndvi, rural_bgr)
    crop_mask = mask_outside_rural_bgr(crop_mask.astype(float), rural_bgr)
    urban_expansion_mask = mask_outside_rural_bgr(urban_expansion_mask.astype(float), rural_bgr)
    landcover_change_mask = mask_outside_rural_bgr(landcover_change_mask.astype(float), rural_bgr)

    ndvi_vmin = np.nanpercentile(mean_ndvi, 2)
    ndvi_vmax = np.nanpercentile(mean_ndvi, 98)

    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.6))
    plot_panel(axes[0], mean_ndvi, 'Original mean NDVI', boundary_edges, bounds, cmap='YlGn', vmin=ndvi_vmin, vmax=ndvi_vmax)
    plot_panel(axes[1], crop_mask, 'Cropland mask', boundary_edges, bounds, cmap='Reds', vmin=0, vmax=1)
    plot_panel(axes[2], urban_expansion_mask, 'IS expansion mask', boundary_edges, bounds, cmap='Reds', vmin=0, vmax=1)
    plot_panel(axes[3], landcover_change_mask, 'Landcover change mask', boundary_edges, bounds, cmap='Reds', vmin=0, vmax=1)
    plot_panel(axes[4], remain_ndvi, 'Remaining NDVI', boundary_edges, bounds, cmap='YlGn', vmin=ndvi_vmin, vmax=ndvi_vmax)

    legend_handles = [
        Line2D([0], [0], color='#810f7c', lw=1.2, label='Urban core'),
        Line2D([0], [0], color='#000000', lw=1.2, label='Urban edge'),
        Line2D([0], [0], color='#e31a1c', lw=1.2, label='Rural background'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False, fontsize=LEGEND_SIZE)
    figToPath = current_dir + '/4_Figures/Rd1_R1_8_p4_figure_plot'
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, wspace=0.35)
    plt.savefig(figToPath, dpi=900)
    # plt.close(fig)

    print(id)
