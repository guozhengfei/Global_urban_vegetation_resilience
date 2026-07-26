import os
import time
import warnings

import cv2
import multiprocess as mp
import numpy as np
import pandas as pd
import tifffile as tf

warnings.filterwarnings("ignore")

ROLLING_YEARS = 4
DESEASON_YEAR_OPTIONS = (7, 8, 9)
BANDS_PER_YEAR = 12
N_MONTHS = 264
EDGE_YEAR_TRIM = 2


def fit_residuals_chunk(chunk):
    x = chunk[:, :, :-1]
    y = chunk[:, :, -1]
    n_pixels, n_time, _ = x.shape
    residuals = np.full((n_pixels, n_time), np.nan, dtype=float)

    finite_rows = ~(np.isnan(x).any(axis=2) | np.isnan(y))
    full_rows = finite_rows.all(axis=1)

    if np.any(full_rows):
        x_full = x[full_rows]
        y_full = y[full_rows]
        design = np.concatenate(
            [np.ones((x_full.shape[0], n_time, 1), dtype=float), x_full],
            axis=2,
        )
        xtx = np.einsum('nti,ntj->nij', design, design)
        xty = np.einsum('nti,nt->ni', design, y_full)
        try:
            beta = np.linalg.solve(xtx, xty[:, :, np.newaxis])[:, :, 0]
        except np.linalg.LinAlgError:
            beta = np.full((x_full.shape[0], 5), np.nan, dtype=float)
            for i in range(x_full.shape[0]):
                beta[i], *_ = np.linalg.lstsq(design[i], y_full[i], rcond=None)

        residuals[full_rows] = (
            y_full
            - beta[:, 2, np.newaxis] * x_full[:, :, 1]
            - beta[:, 3, np.newaxis] * x_full[:, :, 2]
            - beta[:, 4, np.newaxis] * x_full[:, :, 3]
        )

    for idx in np.flatnonzero(~full_rows):
        valid = finite_rows[idx]
        if valid.sum() < 5:
            continue
        design = np.column_stack([np.ones(valid.sum()), x[idx, valid]])
        try:
            beta, *_ = np.linalg.lstsq(design, y[idx, valid], rcond=None)
        except np.linalg.LinAlgError:
            continue
        residuals[idx] = (
            y[idx]
            - beta[2] * x[idx, :, 1]
            - beta[3] * x[idx, :, 2]
            - beta[4] * x[idx, :, 3]
        )

    return residuals


def rolling_ar1_matrix(residuals, rolling_years):
    window = BANDS_PER_YEAR * rolling_years
    output = np.full(residuals.shape, np.nan, dtype=float)
    if residuals.shape[1] < window:
        return output

    windows = np.lib.stride_tricks.sliding_window_view(residuals, window_shape=window, axis=1)
    left = windows[:, :, :-1]
    right = windows[:, :, 1:]
    valid = np.isfinite(left) & np.isfinite(right)
    n = valid.sum(axis=2)

    left_sum = np.where(valid, left, 0).sum(axis=2)
    right_sum = np.where(valid, right, 0).sum(axis=2)
    left_mean = np.divide(left_sum, n, out=np.full_like(left_sum, np.nan, dtype=float), where=n > 0)
    right_mean = np.divide(right_sum, n, out=np.full_like(right_sum, np.nan, dtype=float), where=n > 0)

    left_anom = np.where(valid, left - left_mean[:, :, np.newaxis], 0)
    right_anom = np.where(valid, right - right_mean[:, :, np.newaxis], 0)
    covariance = (left_anom * right_anom).sum(axis=2)
    left_var = (left_anom ** 2).sum(axis=2)
    right_var = (right_anom ** 2).sum(axis=2)

    denominator = np.sqrt(left_var * right_var)
    corr = np.divide(covariance, denominator, out=np.full_like(covariance, np.nan, dtype=float), where=denominator > 0)
    corr[n < window - 1] = np.nan

    start = window // 2
    output[:, start:start + corr.shape[1]] = corr
    return output


def ar1_chunk(args):
    chunk, rolling_years = args
    residuals = fit_residuals_chunk(chunk)
    return rolling_ar1_matrix(residuals, rolling_years)


def iqr_filter(array):
    p25 = np.nanpercentile(array, 25, axis=1)
    p75 = np.nanpercentile(array, 75, axis=1)
    iqr = p75 - p25
    max_v = np.tile(p75 + 0.5 * iqr, (array.shape[1], 1)).T
    min_v = np.tile(p25 - 0.5 * iqr, (array.shape[1], 1)).T
    array[array < min_v] = np.nan
    array[array > max_v] = np.nan
    return array


def fill_nan_with_climatology(evi, bands_year=BANDS_PER_YEAR):
    num_years = evi.shape[1] // bands_year
    climatology = np.zeros((evi.shape[0], evi.shape[1]))

    for year in range(num_years):
        if year < 2:
            start = 0
            end = 5 * bands_year
        elif year > num_years - 3:
            start = (num_years - 5) * bands_year
            end = num_years * bands_year
        else:
            start = (year - 2) * bands_year
            end = (year + 3) * bands_year

        climatology[:, year * bands_year:(year + 1) * bands_year] = np.nanmean(
            evi[:, start:end].reshape(evi.shape[0], 5, bands_year), axis=1
        )

    return np.where(np.isnan(evi), climatology, evi)


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


def calculate_linear_trend(y):
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if np.sum(mask) < 3:
        return np.nan

    x_valid = x[mask]
    y_valid = y[mask]
    try:
        a = np.vstack([x_valid, np.ones(len(x_valid))]).T
        slope, _ = np.linalg.lstsq(a, y_valid, rcond=None)[0]
        return slope
    except Exception:
        return np.nan


def yearly_mean(monthly_data):
    years = int(np.ceil(monthly_data.shape[2] / BANDS_PER_YEAR))
    tac_yearly = np.full(monthly_data.shape[:2] + (years,), np.nan, dtype=float)
    for year in range(years):
        start = year * BANDS_PER_YEAR
        end = min(start + BANDS_PER_YEAR, monthly_data.shape[2])
        tac_yearly[:, :, year] = np.nanmean(monthly_data[:, :, start:end], axis=2)
    return tac_yearly


def load_city_ids(current_dir):
    filenames = os.listdir(os.path.join(current_dir, '1_Input/ta_anom'))
    ids = []
    for name in filenames:
        if not name.endswith('.npy'):
            continue
        ids.append(float(name.split('_')[-1].split('.npy')[0]))
    return np.sort(ids)


def build_pixel_mask(evi0, city_id, folders):
    crop_frc = tf.imread(os.path.join(folders['crop'], 'cropC_' + str(city_id) + '.tif'))
    crop_frc = cv2.resize(crop_frc, (evi0.shape[1], evi0.shape[0]), cv2.INTER_LINEAR)

    grass_frc = tf.imread(os.path.join(folders['grass'], 'grassC_' + str(city_id) + '.tif'))
    grass_frc = cv2.resize(grass_frc, (evi0.shape[1], evi0.shape[0]), cv2.INTER_LINEAR)

    tree_frc = tf.imread(os.path.join(folders['tree'], 'treeC_' + str(city_id) + '.tif'))
    tree_frc = cv2.resize(tree_frc, (evi0.shape[1], evi0.shape[0]), cv2.INTER_LINEAR)

    water_frc = tf.imread(os.path.join(folders['water'], 'waterC_' + str(city_id) + '.tif'))
    water_frc = cv2.resize(water_frc, (evi0.shape[1], evi0.shape[0]), cv2.INTER_LINEAR)

    urban_exp_frc = tf.imread(os.path.join(folders['urban_exp'], 'urban_exp_C_' + str(city_id) + '.tif'))
    if urban_exp_frc.shape != evi0.shape[:2]:
        urban_exp_frc = cv2.resize(urban_exp_frc, (evi0.shape[1], evi0.shape[0]), cv2.INTER_LINEAR)

    crop_frc[crop_frc > 0.2] = np.nan
    veg_nature = grass_frc + tree_frc
    crop_frc[crop_frc > veg_nature] = np.nan
    mask_miss = np.sum(np.isnan(evi0), axis=2) / evi0.shape[2]

    return np.isnan(crop_frc) | (mask_miss > 0.3) | (water_frc > 0.3) | (urban_exp_frc > 0.2)


def preprocess_ndvi(city_id, folders, yr_length):
    evi0 = tf.imread(os.path.join(folders['evi'], 'nadir_ndvi_15d_' + str(city_id) + '.tif'))[:, :, :N_MONTHS * 2][:, :, ::2]
    base_mask = build_pixel_mask(evi0, city_id, folders)
    base_mask_1d = base_mask.flatten()

    evi_rsp = evi0.reshape((evi0.shape[0] * evi0.shape[1], evi0.shape[2]))
    evi = evi_rsp[~base_mask_1d, :]
    evi = iqr_filter(evi)
    evi = fill_nan_with_climatology(evi)

    if evi.shape[0] < 10:
        return None

    evi_yr = np.zeros_like(evi)
    yr_num = N_MONTHS // BANDS_PER_YEAR
    for year in range(yr_num):
        start = year * BANDS_PER_YEAR
        end = start + BANDS_PER_YEAR
        evi_year = np.nanmean(evi[:, start:end], axis=1)
        evi_yr[:, start:end] = np.repeat(evi_year[:, np.newaxis], BANDS_PER_YEAR, axis=1)

    rm_offline = evi - evi_yr
    del evi_yr, evi

    evi_sea_rep = np.zeros_like(rm_offline)
    for year in range(yr_num):
        half_window = yr_length // 2
        if yr_length % 2 == 0:
            start = max((year - half_window) * BANDS_PER_YEAR, 0)
            end = min((year + half_window) * BANDS_PER_YEAR, BANDS_PER_YEAR * yr_num)
        else:
            start = max((year - half_window) * BANDS_PER_YEAR, 0)
            end = min((year + half_window + 1) * BANDS_PER_YEAR, BANDS_PER_YEAR * yr_num)
        data_i = rm_offline[:, start:end]
        evi_sea = np.nanmean(
            np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / BANDS_PER_YEAR), BANDS_PER_YEAR)),
            axis=1,
        )
        evi_sea_rep[:, year * BANDS_PER_YEAR:(year + 1) * BANDS_PER_YEAR] = evi_sea

    res_ndvi = rm_offline - evi_sea_rep
    res_ndvi[np.isnan(res_ndvi)] = 0

    return evi0, evi_rsp, base_mask_1d, res_ndvi


def build_model_array(city_id, folders, base_mask_1d, res_ndvi):
    res_pr_org = np.load(os.path.join(folders['pre'], 'pr_month_anomaly_' + str(city_id) + '.npy'))[:, :, 1:1 + N_MONTHS]
    res_pr = res_pr_org.reshape((res_pr_org.shape[0] * res_pr_org.shape[1], res_pr_org.shape[2]))[~base_mask_1d, :]
    nan_frac_pr = np.sum(np.isnan(res_pr), axis=1) / res_pr.shape[1]

    res_rad_org = np.load(os.path.join(folders['rad'], 'rad_month_anomaly_' + str(city_id) + '.npy'))[:, :, 1:1 + N_MONTHS]
    res_rad = res_rad_org.reshape((res_rad_org.shape[0] * res_rad_org.shape[1], res_rad_org.shape[2]))[~base_mask_1d, :]
    nan_frac_rad = np.sum(np.isnan(res_rad), axis=1) / res_rad.shape[1]

    res_ta_org = np.load(os.path.join(folders['ta'], 'ta_month_anomaly_' + str(city_id) + '.npy'))[:, :, 1:1 + N_MONTHS]
    res_ta = res_ta_org.reshape((res_ta_org.shape[0] * res_ta_org.shape[1], res_ta_org.shape[2]))[~base_mask_1d, :]
    nan_frac_ta = np.sum(np.isnan(res_ta), axis=1) / res_ta.shape[1]

    nan_frac_ndvi = np.sum(np.isnan(res_ndvi), axis=1) / res_ndvi.shape[1]
    invalid_rows = (nan_frac_ndvi > 0.75) | (nan_frac_pr > 0.3) | (nan_frac_rad > 0.3) | (nan_frac_ta > 0.3)

    variables_all = np.stack(
        (res_ndvi[:, :-1], res_ta[:, 1:], res_rad[:, 1:], res_pr[:, 1:], res_ndvi[:, 1:]),
        axis=2,
    )
    return variables_all[~invalid_rows, :, :], invalid_rows


def compute_pixel_tac_yearly(city_id, folders, workers, chunk_size, yr_length):
    preprocessed = preprocess_ndvi(city_id, folders, yr_length)
    if preprocessed is None:
        return None
    evi0, evi_rsp, base_mask_1d, res_ndvi = preprocessed

    variables_all, invalid_rows = build_model_array(city_id, folders, base_mask_1d, res_ndvi)
    if variables_all.shape[0] < 10:
        return None

    chunks = [
        variables_all[start:start + chunk_size]
        for start in range(0, variables_all.shape[0], chunk_size)
    ]
    work_items = [(chunk, ROLLING_YEARS) for chunk in chunks]
    with mp.Pool(workers) as pool:
        chunk_results = list(pool.map(ar1_chunk, work_items))

    valid_pixel_indices = np.flatnonzero(~base_mask_1d)
    valid_pixel_indices = valid_pixel_indices[~invalid_rows]

    ar1_res = np.vstack(chunk_results)
    tac_map_flat = np.full((evi_rsp.shape[0], evi0.shape[2] - 1), np.nan, dtype=float)
    tac_map_flat[valid_pixel_indices, :] = ar1_res
    tac_monthly = tac_map_flat.reshape((evi0.shape[0], evi0.shape[1], evi0.shape[2] - 1))
    tac_yearly = yearly_mean(tac_monthly)

    if tac_yearly.shape[2] > EDGE_YEAR_TRIM * 2:
        tac_yearly = tac_yearly[:, :, EDGE_YEAR_TRIM:-EDGE_YEAR_TRIM]

    return tac_yearly


def aggregate_city_zones(city_id, tac_yearly, folders):
    urban_2018 = tf.imread(os.path.join(folders['urban_2018'], 'fvc_' + str(city_id) + '.tif')).astype(float)
    urban_2018_rsz = cv2.resize(urban_2018, (tac_yearly.shape[1], tac_yearly.shape[0]), cv2.INTER_NEAREST)
    urban_2018_rsz[urban_2018_rsz != float(city_id)] = np.nan

    urban_1990 = tf.imread(os.path.join(folders['urban_1990'], 'urban_1990_' + str(city_id) + '.tif')).astype(float)
    urban_1990 = cv2.resize(urban_1990, (tac_yearly.shape[1], tac_yearly.shape[0]), cv2.INTER_NEAREST)
    urban_1990[urban_1990 == 0] = np.nan

    rural_near = urban_2018_rsz * 1
    for _ in range(3 * 2):
        rural_near = extend_edge(rural_near)

    rural_bgr = rural_near * 1
    for _ in range(10 * 2):
        rural_bgr = extend_edge(rural_bgr)

    urban_1990_3d = np.tile(urban_1990[:, :, np.newaxis], tac_yearly.shape[2])
    tac_urban_core_all = tac_yearly + urban_1990_3d - urban_1990_3d
    if np.sum(~np.isnan(tac_urban_core_all)) < 100:
        return None
    tac_urban_core = np.nanmean(np.nanmean(tac_urban_core_all, axis=0), axis=0)

    urban_1990_v2 = urban_1990 * 1
    urban_1990_v2[np.isnan(urban_1990_v2)] = 0
    urban_1990_v2[urban_1990_v2 != 0] = np.nan
    urban_1990_2018 = rural_near + urban_1990_v2
    urban_1990_2018_3d = np.tile(urban_1990_2018[:, :, np.newaxis], tac_yearly.shape[2])
    tac_urban_edge_all = tac_yearly + urban_1990_2018_3d - urban_1990_2018_3d
    if np.sum(~np.isnan(tac_urban_edge_all)) < 100:
        return None
    tac_urban_edge = np.nanmean(np.nanmean(tac_urban_edge_all, axis=0), axis=0)

    rural_near_v2 = rural_near * 1
    rural_near_v2[np.isnan(rural_near_v2)] = 0
    rural_near_v2[rural_near_v2 != 0] = np.nan
    urban_rural_bgr = rural_bgr + rural_near_v2
    urban_rural_bgr_3d = np.tile(urban_rural_bgr[:, :, np.newaxis], tac_yearly.shape[2])
    tac_rural_all = tac_yearly + urban_rural_bgr_3d - urban_rural_bgr_3d
    if np.sum(~np.isnan(tac_rural_all)) < 100:
        return None
    tac_rural_bgr = np.nanmean(np.nanmean(tac_rural_all, axis=0), axis=0)

    return [tac_urban_core, tac_urban_edge, tac_rural_bgr]


def calculate_trend_table(tac_arr, ids):
    tac_trends = np.full((tac_arr.shape[0], 3), np.nan, dtype=float)
    for city_idx in range(tac_arr.shape[0]):
        for region_idx in range(3):
            tac_trends[city_idx, region_idx] = calculate_linear_trend(tac_arr[city_idx, region_idx])

    df_trends = pd.DataFrame(
        tac_trends,
        columns=['urban_core_trend', 'urban_edge_trend', 'rural_trend'],
    )
    df_trends['ID'] = ids
    df_trends['urban_rural_diff'] = df_trends['urban_core_trend'] - df_trends['rural_trend']
    return df_trends


if __name__ == '__main__':
    start_time = time.time()
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    workers = int(os.environ.get('TAC_WORKERS', min(32, os.cpu_count() or 1)))
    chunk_size = int(os.environ.get('TAC_CHUNK_SIZE', 1024))

    folders = {
        'evi': current_dir + '/1_Input/nadir_ndvi_15d_500/',
        'grass': current_dir + '/1_Input/grassCover_250m/',
        'tree': current_dir + '/1_Input/treeCover_250m/',
        'crop': current_dir + '/1_Input/cropCover_250m/',
        'water': current_dir + '/1_Input/waterCover_250m/',
        'urban_exp': current_dir + '/1_Input/urban_expansion_frac_500m/',
        'pre': current_dir + '/1_Input/pre_anom/',
        'rad': current_dir + '/1_Input/rad_anom/',
        'ta': current_dir + '/1_Input/ta_anom/',
        'urban_2018': current_dir + '/1_Input/urban/urban_2018_1000m/',
        'urban_1990': current_dir + '/1_Input/urban/urban_1990_250m/',
    }

    print('Workers:', workers)
    print('Chunk size:', chunk_size)
    print('Rolling window:', ROLLING_YEARS, 'years')
    print('Deseason windows:', DESEASON_YEAR_OPTIONS)

    city_ids = load_city_ids(current_dir)
    for yr_length in DESEASON_YEAR_OPTIONS:
        run_start_time = time.time()
        print('Starting deseason window:', yr_length, 'years')

        tac_by_city = []
        ids_num = []
        for city_id in city_ids:
            try:
                tac_yearly = compute_pixel_tac_yearly(city_id, folders, workers, chunk_size, yr_length)
                if tac_yearly is None:
                    print(city_id, 'skipped: insufficient valid pixels')
                    continue

                city_tac = aggregate_city_zones(city_id, tac_yearly, folders)
                if city_tac is None:
                    print(city_id, 'skipped: insufficient zone pixels')
                    continue

                tac_by_city.append(city_tac)
                ids_num.append(float(city_id))
                print(city_id)
            except Exception as exc:
                print(city_id, 'failed:', exc)

        if tac_by_city:
            tac_arr = np.array(tac_by_city)
        else:
            tac_arr = np.empty((0, 3, 0), dtype=float)
        ids_arr = np.array(ids_num)
        suffix = f'deseason{yr_length}yr'

        tac_output = current_dir + f'/2_Output/tac_nadir_city_3zones_{suffix}.npz'
        np.savez(tac_output, array1=tac_arr, array2=ids_arr)
        print('Saved:', tac_output)

        trends = calculate_trend_table(tac_arr, ids_arr)
        trend_output = current_dir + f'/2_Output/tac_trends_modis_{suffix}.csv'
        trends.to_csv(trend_output, index=False)
        print('Saved:', trend_output)
        print('Finished deseason window %d years in %.1f s' % (yr_length, time.time() - run_start_time))

    print('Done in %.1f s' % (time.time() - start_time))
