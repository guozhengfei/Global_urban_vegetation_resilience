import os

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import cv2
import tifffile as tf

plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
plt.close()

AXIS_LABEL_SIZE = 14+1
TICK_LABEL_SIZE = 12+1
PANEL_LABEL_SIZE = 14+1
ANNOTATION_SIZE = 12+1


def load_tac_3zone_df(path):
    data = np.load(path)
    tac = data['array1']
    city_id = data['array2']
    df = pd.DataFrame(tac, columns=['urban_core', 'urban_edge', 'rural_bgr'])
    df['ID'] = city_id
    df['urban_rural_tac_diff'] = df['urban_core'] - df['rural_bgr']
    return df


def extend_edge(array):
    array1 = array * 1
    array1[1:, :] = array[:-1, :]
    array2 = array * 1
    array2[:-1, :] = array[1:, :]
    array3 = array * 1
    array3[:, 1:] = array[:, :-1]
    array4 = array * 1
    array4[:, :-1] = array[:, 1:]
    array5 = array * 1
    array5[:-1, 1:] = array[1:, :-1]
    array6 = array * 1
    array6[:-1, :-1] = array[1:, 1:]
    array7 = array * 1
    array7[1:, 1:] = array[:-1, :-1]
    array8 = array * 1
    array8[1:, :-1] = array[:-1, 1:]
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


def resize_nearest(array, target_shape):
    if array.shape == target_shape:
        return array
    return cv2.resize(array, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


def resize_linear(array, target_shape):
    if array.shape == target_shape:
        return array
    return cv2.resize(array, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def managed_pixel_mask(managed_grass_file, managed_tree_file, threshold=0.5):
    managed_grass = tf.imread(managed_grass_file).astype(float)
    managed_tree = tf.imread(managed_tree_file).astype(float)
    managed_tree = resize_linear(managed_tree, managed_grass.shape)
    return (managed_grass > threshold) | (managed_tree > threshold)


def build_zone_mask(urban_2018, urban_1990, city_id, target_shape):
    urban_2018_rsz = resize_nearest(urban_2018, target_shape).astype(float)
    urban_2018_rsz[urban_2018_rsz != float(city_id)] = np.nan

    urban_1990 = resize_nearest(urban_1990, target_shape).astype(float)
    urban_1990[urban_1990 == 0] = np.nan
    urban_1990[~np.isnan(urban_1990)] = 1

    rural_near = urban_2018_rsz * 1
    rural_near = extend_edge(extend_edge(rural_near))

    rural_bgr = rural_near * 1
    rural_bgr = extend_edge_1(rural_bgr, 20)
    rural_bgr = extend_edge(extend_edge(extend_edge(extend_edge(extend_edge(rural_bgr)))))
    rural_bgr[~np.isnan(rural_near)] = 1

    rural_bgr_mask = ~np.isnan(rural_bgr)
    return (~np.isnan(urban_1990)) | (~np.isnan(rural_near)) | rural_bgr_mask


def list_city_ids(managed_grass_folder):
    city_ids = []
    for filename in os.listdir(managed_grass_folder):
        if not filename.startswith('managed_grassland_fraction_') or not filename.endswith('.tif'):
            continue
        if filename.startswith('._'):
            continue
        city_id = filename.replace('managed_grassland_fraction_', '').replace('.tif', '')
        city_ids.append(city_id)
    return sorted(city_ids, key=lambda value: float(value))


def calculate_managed_fraction_by_city(
    managed_grass_folder,
    managed_tree_folder,
    urban_folder_2018,
    urban_folder_1990,
    output_file,
    threshold=0.5,
):
    results = []
    for city_id in list_city_ids(managed_grass_folder):
        managed_grass_file = os.path.join(managed_grass_folder, 'managed_grassland_fraction_' + city_id + '.tif')
        managed_tree_file = os.path.join(managed_tree_folder, 'managedTreeC_' + city_id + '.tif')
        urban_2018_file = os.path.join(urban_folder_2018, 'urban_' + city_id + '.tif')
        urban_1990_file = os.path.join(urban_folder_1990, 'urban_1990_' + city_id + '.tif')

        if not os.path.exists(managed_tree_file):
            results.append({
                'ID': city_id,
                'managed_pixels': np.nan,
                'zone_pixels': np.nan,
                'managed_pixel_fraction': np.nan,
                'status': 'missing managed tree raster',
            })
            continue

        if not os.path.exists(urban_2018_file) or not os.path.exists(urban_1990_file):
            results.append({
                'ID': city_id,
                'managed_pixels': np.nan,
                'zone_pixels': np.nan,
                'managed_pixel_fraction': np.nan,
                'status': 'missing urban raster',
            })
            continue

        try:
            managed_mask = managed_pixel_mask(managed_grass_file, managed_tree_file, threshold=threshold)
            urban_2018 = tf.imread(urban_2018_file).astype(float)
            urban_1990 = tf.imread(urban_1990_file).astype(float)
            zone_mask = build_zone_mask(urban_2018, urban_1990, city_id, managed_mask.shape)

            zone_pixels = int(np.sum(zone_mask))
            managed_pixels = int(np.sum(managed_mask & zone_mask))
            fraction = managed_pixels / zone_pixels if zone_pixels > 0 else np.nan
            status = 'ok' if zone_pixels > 0 else 'empty zone'
        except Exception as exc:
            managed_pixels = np.nan
            zone_pixels = np.nan
            fraction = np.nan
            status = f'error: {exc}'

        results.append({
            'ID': city_id,
            'managed_pixels': managed_pixels,
            'zone_pixels': zone_pixels,
            'managed_pixel_fraction': fraction,
            'status': status,
        })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    return df


def panel_managed_histogram(ax, csv_path):
    df = pd.read_csv(csv_path)
    values = df.loc[df['status'].eq('ok'), 'managed_pixel_fraction'].dropna().to_numpy()

    ax.hist(values, bins=np.linspace(0, 0.4, 25), color='0.70', edgecolor='0.25', linewidth=0.6)
    ax.axvline(np.nanmean(values), color='#b2182b', lw=1.2)
    ax.set_xlabel('Managed pixel fraction', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('City count', fontsize=AXIS_LABEL_SIZE)
    ax.text(
        0.96, 0.94,
        f'mean = {np.nanmean(values):.2f}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=ANNOTATION_SIZE
    )


def panel_tac_bar(ax, tac_path):
    tac = np.load(tac_path)['array1']
    tac_mean = np.nanmean(tac, axis=0)
    tac_err = np.nanstd(tac, axis=0) * 0.25

    bars = ax.bar(
        np.arange(3),
        tac_mean,
        yerr=tac_err,
        width=0.36,
        color=['#2166ac', '#67a9cf', '#b2182b'],
        error_kw={'lw': 0.8, 'capsize': 2, 'capthick': 0.8}
    )

    ax.set_xticks([0, 1, 2], ['UC', 'UE', 'RA'])
    ax.set_ylabel('No-managed TAC', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim([0.10, 0.22])


def panel_urban_rural_regression(ax, tac_path):
    tac = np.load(tac_path)['array1']
    mask = ~np.isnan(tac[:, 0]) & ~np.isnan(tac[:, 2])
    x = tac[mask, 0]
    y = tac[mask, 2]

    above = y > x
    below = ~above
    ax.plot(x[above], y[above], 'o', mfc='none', color='C0', ms=4, mew=0.8)
    ax.plot(x[below], y[below], 'o', mfc='none', color='C3', ms=4, mew=0.8)

    axis_min = min(np.nanmin(x), np.nanmin(y))
    axis_max = max(np.nanmax(x), np.nanmax(y))
    padding = (axis_max - axis_min) * 0.08
    axis_min -= padding
    axis_max += padding

    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7, lw=0.9)
    fit = np.poly1d(np.polyfit(x, y, 1))
    x_fit = np.linspace(axis_min, axis_max, 100)
    ax.plot(x_fit, fit(x_fit), 'k-', alpha=0.7, lw=0.9)

    reg = st.linregress(x, y)
    ax.text(
        0.05, 0.95,
        f'r = {reg.rvalue:.2f}',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=ANNOTATION_SIZE
    )
    ax.set_xlabel('No-managed TAC$_{UC}$', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('No-managed TAC$_{RA}$', fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal', adjustable='box')


def panel_tac_diff_scatter(ax, original_path, no_managed_path):
    df_original = load_tac_3zone_df(original_path)
    df_no_managed = load_tac_3zone_df(no_managed_path)
    df_compare = df_original[['ID', 'urban_rural_tac_diff']].merge(
        df_no_managed[['ID', 'urban_rural_tac_diff']],
        on='ID',
        suffixes=('_original', '_no_managed')
    )
    df_compare = df_compare.dropna(subset=['urban_rural_tac_diff_original', 'urban_rural_tac_diff_no_managed'])

    x = df_compare['urban_rural_tac_diff_original'].to_numpy()
    y = df_compare['urban_rural_tac_diff_no_managed'].to_numpy()
    r, _ = st.pearsonr(x, y)
    rmse = np.sqrt(np.nanmean((y - x) ** 2))

    ax.scatter(x, y, s=14, color='#2166ac', alpha=0.65, edgecolors='none')

    axis_min = np.nanmin([np.nanmin(x), np.nanmin(y)])
    axis_max = np.nanmax([np.nanmax(x), np.nanmax(y)])
    padding = (axis_max - axis_min) * 0.08
    axis_min -= padding
    axis_max += padding
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color='0.25', lw=0.9, ls='--')

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Original ΔTAC', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel('No-managed ΔTAC', fontsize=AXIS_LABEL_SIZE)
    ax.text(
        0.05, 0.95,
        f'r = {r:.2f}\nRMSE = {rmse:.3f}',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=ANNOTATION_SIZE
    )


def format_panel(ax, label):
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, ha='left', va='top',
            fontsize=PANEL_LABEL_SIZE, fontweight='bold')
    ax.tick_params(width=0.8, labelsize=TICK_LABEL_SIZE)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    managed_grass_folder = current_dir + '/1_Input/Managed_grasslandFraction_500m'
    managed_tree_folder = current_dir + '/1_Input/managedTreeCover_500m'
    urban_folder_2018 = current_dir + '/1_Input/urban/urban_2018_250m'
    urban_folder_1990 = current_dir + '/1_Input/urban/urban_1990_250m'
    no_managed_tac_path = current_dir + '/2_Output/tac_nadir_city_3zones_no_managed_modis.npz'
    original_tac_path = current_dir + '/2_Output/tac_nadir_city_3zones_all.npz'
    managed_fraction_path = current_dir + '/2_Output/managed_pixel_fraction_by_city.csv'
    fig_to_path = current_dir + '/4_Figures/Rd1_R2_7_tac_no_managed_four_panel_MODIS'

    calculate_managed_fraction_by_city(
        managed_grass_folder,
        managed_tree_folder,
        urban_folder_2018,
        urban_folder_1990,
        managed_fraction_path,
        threshold=0.5,
    )

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.4))
    panel_managed_histogram(axes[0], managed_fraction_path)
    panel_tac_bar(axes[1], no_managed_tac_path)
    panel_urban_rural_regression(axes[2], no_managed_tac_path)
    panel_tac_diff_scatter(axes[3], original_tac_path, no_managed_tac_path)

    for ax, label in zip(axes, ['', '', '', '']):
        format_panel(ax, label)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(fig_to_path, dpi=900)
