import os
import re
import warnings

import cv2
import numpy as np
import pandas as pd
import rasterio

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")


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


def list_city_csvs(vi_landsat_folder):
    pattern = re.compile(r'^NDVI_8d_([0-9]+(?:\.[0-9]+)?)\.csv$')
    city_files = []
    for filename in os.listdir(vi_landsat_folder):
        match = pattern.match(filename)
        if match:
            city_files.append((float(match.group(1)), filename))
    return sorted(city_files, key=lambda item: item[0])


def resolve_tac_path(city_id, vi_landsat_folder, tac_fallback_folder):
    filename = f'tac_{city_id}.npy'
    vi_path = os.path.join(vi_landsat_folder, filename)
    if os.path.exists(vi_path):
        return vi_path

    fallback_path = os.path.join(tac_fallback_folder, filename)
    if os.path.exists(fallback_path):
        return fallback_path

    return vi_path


def sample_raster_values(raster_path, lon, lat):
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(float)
        rows, cols = rasterio.transform.rowcol(src.transform, lon, lat)

    rows = np.asarray(rows)
    cols = np.asarray(cols)
    valid = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    values = np.full(len(lon), np.nan, dtype=float)
    values[valid] = data[rows[valid], cols[valid]]
    return values


def build_zone_masks(city_id, urban_2018_path, urban_1990_path):
    with rasterio.open(urban_2018_path) as src:
        urban_2018 = src.read(1).astype(float)
        transform = src.transform
    with rasterio.open(urban_1990_path) as src:
        urban_1990 = src.read(1).astype(float)

    urban_2018[urban_2018 != float(city_id)] = np.nan
    urban_1990[urban_1990 == 0] = np.nan
    urban_1990[~np.isnan(urban_1990)] = 1

    rural_near = extend_edge(extend_edge(urban_2018))
    rural_bgr = rural_near * 1
    rural_bgr = extend_edge_1(rural_bgr, 20)
    rural_bgr = extend_edge(extend_edge(extend_edge(extend_edge(extend_edge(rural_bgr)))))
    rural_bgr[~np.isnan(rural_near)] = 1
    rural_bgr[~np.isnan(rural_near)] = np.nan

    return ~np.isnan(urban_1990), ~np.isnan(rural_bgr), transform


def sample_zone_masks(lon, lat, urban_core_mask, rural_bgr_mask, transform):
    rows, cols = rasterio.transform.rowcol(transform, lon, lat)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    valid = (
        (rows >= 0) & (rows < urban_core_mask.shape[0]) &
        (cols >= 0) & (cols < urban_core_mask.shape[1])
    )

    urban_core = np.zeros(len(lon), dtype=bool)
    rural_bgr = np.zeros(len(lon), dtype=bool)
    urban_core[valid] = urban_core_mask[rows[valid], cols[valid]]
    rural_bgr[valid] = rural_bgr_mask[rows[valid], cols[valid]]
    return urban_core, rural_bgr


def mean_tac_diff(pixel_df, tac_col, mask):
    subset = pixel_df[mask]
    core = subset.loc[subset['zone'] == 'urban_core', tac_col]
    rural = subset.loc[subset['zone'] == 'rural_bgr', tac_col]

    core_n = int(core.notna().sum())
    rural_n = int(rural.notna().sum())
    core_mean = core.mean(skipna=True) if core_n > 0 else np.nan
    rural_mean = rural.mean(skipna=True) if rural_n > 0 else np.nan
    return core_mean, rural_mean, core_mean - rural_mean, core_n, rural_n


def process_city(city_id, csv_filename, paths):
    csv_path = os.path.join(paths['vi_landsat_folder'], csv_filename)
    tac_path = resolve_tac_path(city_id, paths['vi_landsat_folder'], paths['tac_fallback_folder'])
    tree_path = os.path.join(paths['tree_folder'], f'treeC_{city_id}.tif')
    grass_path = os.path.join(paths['grass_folder'], f'grassC_{city_id}.tif')
    urban_2018_path = os.path.join(paths['urban_2018_folder'], f'urban_{city_id}.tif')
    urban_1990_path = os.path.join(paths['urban_1990_folder'], f'urban_1990_{city_id}.tif')

    required = [csv_path, tac_path, tree_path, grass_path, urban_2018_path, urban_1990_path]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        return None, {'ID': city_id, 'status': 'missing file: ' + '; '.join(missing)}

    city_df = pd.read_csv(csv_path)
    lon = city_df.iloc[:, -2].astype(float).to_numpy()
    lat = city_df.iloc[:, -1].astype(float).to_numpy()
    tac = np.load(tac_path)

    if tac.shape[0] != len(city_df):
        return None, {
            'ID': city_id,
            'status': f'row mismatch: csv={len(city_df)}, tac={tac.shape[0]}',
        }

    tree_frac = sample_raster_values(tree_path, lon, lat)
    grass_frac = sample_raster_values(grass_path, lon, lat)
    urban_core_mask, rural_bgr_mask, zone_transform = build_zone_masks(city_id, urban_2018_path, urban_1990_path)
    urban_core, rural_bgr = sample_zone_masks(lon, lat, urban_core_mask, rural_bgr_mask, zone_transform)

    pixel_df = pd.DataFrame({
        'ID': city_id,
        'pixel_index': np.arange(len(city_df)),
        'lon': lon,
        'lat': lat,
        'tree_frac': tree_frac,
        'grass_frac': grass_frac,
        'dominance': np.where(tree_frac > grass_frac, 'tree_dominant',
                              np.where(grass_frac > tree_frac, 'grass_dominant', 'mixed_equal')),
        'zone': np.where(urban_core, 'urban_core', np.where(rural_bgr, 'rural_bgr', 'other')),
        'tac_mean': np.nanmean(tac, axis=1),
    })

    all_core, all_rural, all_diff, all_core_n, all_rural_n = mean_tac_diff(pixel_df, 'tac_mean', pixel_df['dominance'].notna())
    tree_core, tree_rural, tree_diff, tree_core_n, tree_rural_n = mean_tac_diff(
        pixel_df, 'tac_mean', pixel_df['dominance'] == 'tree_dominant'
    )
    grass_core, grass_rural, grass_diff, grass_core_n, grass_rural_n = mean_tac_diff(
        pixel_df, 'tac_mean', pixel_df['dominance'] == 'grass_dominant'
    )

    summary = {
        'ID': city_id,
        'all_core_tac_mean': all_core,
        'all_rural_tac_mean': all_rural,
        'all_urban_rural_tac_diff': all_diff,
        'all_core_n': all_core_n,
        'all_rural_n': all_rural_n,
        'tree_core_tac_mean': tree_core,
        'tree_rural_tac_mean': tree_rural,
        'tree_urban_rural_tac_diff': tree_diff,
        'tree_core_n': tree_core_n,
        'tree_rural_n': tree_rural_n,
        'grass_core_tac_mean': grass_core,
        'grass_rural_tac_mean': grass_rural,
        'grass_urban_rural_tac_diff': grass_diff,
        'grass_core_n': grass_core_n,
        'grass_rural_n': grass_rural_n,
        'status': 'ok',
    }
    return pixel_df, summary


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    paths = {
        'vi_landsat_folder': current_dir + '/2_Output/VI_Landsat',
        'tac_fallback_folder': current_dir + '/2_Output/tac_Landsat',
        'tree_folder': current_dir + '/1_Input/treeCover_250m',
        'grass_folder': current_dir + '/1_Input/grassCover_250m',
        'urban_2018_folder': current_dir + '/1_Input/urban/urban_2018_250m',
        'urban_1990_folder': current_dir + '/1_Input/urban/urban_1990_250m',
    }

    pixel_outputs = []
    summaries = []
    for city_id, csv_filename in list_city_csvs(paths['vi_landsat_folder']):
        pixel_df, summary = process_city(city_id, csv_filename, paths)
        summaries.append(summary)
        if pixel_df is not None:
            pixel_outputs.append(pixel_df)
        print(city_id, summary['status'])

    summary_df = pd.DataFrame(summaries)
    summary_path = current_dir + '/2_Output/VI_Landsat/tree_grass_dominant_urban_rural_tac_diff_by_city.csv'
    summary_df.to_csv(summary_path, index=False)
    print('Saved:', summary_path)

    if pixel_outputs:
        pixel_df = pd.concat(pixel_outputs, ignore_index=True)
        pixel_path = current_dir + '/2_Output/VI_Landsat/tree_grass_fraction_zone_by_landsat_pixel.csv'
        pixel_df.to_csv(pixel_path, index=False)
        print('Saved:', pixel_path)
