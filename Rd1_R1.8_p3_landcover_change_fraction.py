import os
import re
import warnings

import cv2
import numpy as np
import pandas as pd
import tifffile as tf

warnings.filterwarnings("ignore")


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


def landcover_change_mask(landcover_file):
    landcover = tf.imread(landcover_file)
    if landcover.ndim != 3:
        raise ValueError(f'Expected a 3-band land cover tif: {landcover_file}')
    if landcover.shape[0] == 3 and landcover.shape[-1] != 3:
        landcover = np.moveaxis(landcover, 0, -1)
    if landcover.shape[-1] != 3:
        raise ValueError(f'Expected exactly 3 land cover bands: {landcover_file}')

    return landcover[:, :, 1] != landcover[:, :, 2]


def build_fig01g_zone_masks(urban_2018, urban_1990, city_id, target_shape):
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
    zone_mask = (~np.isnan(urban_1990)) | (~np.isnan(rural_near)) | rural_bgr_mask
    return zone_mask, rural_bgr_mask


def clip_lcc_to_rural_bgr(lcc_mask, rural_bgr_mask):
    if lcc_mask.shape != rural_bgr_mask.shape:
        raise ValueError(f'LCC and rural_bgr shape mismatch: {lcc_mask.shape} vs {rural_bgr_mask.shape}')

    clipped = lcc_mask.copy()
    clipped[~rural_bgr_mask] = False
    return clipped


def assert_same_resolution(lcc_mask, zone_mask, rural_bgr_mask):
    if lcc_mask.shape != zone_mask.shape:
        raise ValueError(f'LCC and zone shape mismatch: {lcc_mask.shape} vs {zone_mask.shape}')
    if lcc_mask.shape != rural_bgr_mask.shape:
        raise ValueError(f'LCC and rural_bgr shape mismatch: {lcc_mask.shape} vs {rural_bgr_mask.shape}')


def list_city_ids(landcover_folder):
    pattern = re.compile(r'^landcover_500m_([0-9]+(?:\.[0-9]+)?)\.tif$')
    city_ids = []
    for filename in os.listdir(landcover_folder):
        match = pattern.match(filename)
        if match:
            city_ids.append(match.group(1))
    return sorted(city_ids, key=lambda value: float(value))


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    landcover_folder = current_dir + '/1_Input/landcover_modis_500m/'
    urban_folder_2018 = current_dir + '/1_Input/urban/urban_2018_250m/'
    urban_folder_1990 = current_dir + '/1_Input/urban/urban_1990_250m/'
    output_file = current_dir + '/2_Output/landcover_change_fraction_by_city.csv'

    results = []
    for city_id in list_city_ids(landcover_folder):
        landcover_file = landcover_folder + 'landcover_500m_' + city_id + '.tif'
        urban_2018_file = urban_folder_2018 + 'urban_' + city_id + '.tif'
        urban_1990_file = urban_folder_1990 + 'urban_1990_' + city_id + '.tif'

        if not os.path.exists(urban_2018_file) or not os.path.exists(urban_1990_file):
            results.append({
                'ID': city_id,
                'landcover_change_pixels': np.nan,
                'zone_pixels': np.nan,
                'landcover_change_fraction': np.nan,
                'status': 'missing urban raster',
            })
            continue

        try:
            lcc_mask = landcover_change_mask(landcover_file)
            target_shape = lcc_mask.shape
            urban_2018 = tf.imread(urban_2018_file).astype(float)
            urban_1990 = tf.imread(urban_1990_file).astype(float)
            zone_mask, rural_bgr_mask = build_fig01g_zone_masks(urban_2018, urban_1990, city_id, target_shape)
            assert_same_resolution(lcc_mask, zone_mask, rural_bgr_mask)
            lcc_mask = clip_lcc_to_rural_bgr(lcc_mask, rural_bgr_mask)

            zone_pixels = int(np.sum(zone_mask))
            lcc_pixels = int(np.sum(lcc_mask & zone_mask))
            fraction = lcc_pixels / zone_pixels if zone_pixels > 0 else np.nan
            status = 'ok' if zone_pixels > 0 else 'empty zone'
        except Exception as exc:
            zone_pixels = np.nan
            lcc_pixels = np.nan
            fraction = np.nan
            status = f'error: {exc}'

        results.append({
            'ID': city_id,
            'landcover_change_pixels': lcc_pixels,
            'zone_pixels': zone_pixels,
            'landcover_change_fraction': fraction,
            'status': status,
        })
        print(city_id, status, fraction)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print('Saved:', output_file)
