import numpy as np
import tifffile as tf
# import matplotlib.pyplot as plt
# import matplotlib; matplotlib.use('Qt5Agg')
import scipy.signal as ss
import multiprocess as mp
import os
import warnings
import numba as nb

warnings.filterwarnings("ignore")
import cv2

def fill_with_climatology(vis,yr_num,bands_year):
    Evi_sea_rep = np.zeros_like(vis)
    for yr in range(yr_num):
        start_index = (yr - 2) * bands_year
        if start_index < 0: start_index = 0
        end_index = (yr + 3) * bands_year
        if end_index > bands_year * yr_num: end_index = bands_year * yr_num
        data_i = vis[:, start_index:end_index]
        Evi_sea = np.nanmean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)),
                             axis=1)
        Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea
    vis[np.isnan(vis)] = Evi_sea_rep[np.isnan(vis)]
    return vis

import pandas as pd
import numpy.ma as ma
from statsmodels.tsa.seasonal import STL

@nb.njit(parallel=True)
def sliding_ar1(res, window_size, min_periods):
    n = len(res)
    ar1 = np.full(n, np.nan)
    valid_window = window_size - 1  # 因为计算的是x和x+1的相关性

    for i in nb.prange(n):
        # 计算中心对齐的窗口
        half = window_size // 2
        start = max(0, i - half)
        end = min(n, i + half + 1)

        if (end - start) < valid_window:
            continue

        valid_count = 0
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0

        for j in range(start, end - 1):
            x = res[j]
            y = res[j + 1]
            if not np.isnan(x) and not np.isnan(y):
                sum_x += x
                sum_y += y
                sum_xy += x * y
                sum_x2 += x * x
                sum_y2 += y * y
                valid_count += 1

        if valid_count >= min_periods and valid_count > 1:
            cov = (sum_xy - sum_x * sum_y / valid_count)
            var_x = sum_x2 - sum_x ** 2 / valid_count
            var_y = sum_y2 - sum_y ** 2 / valid_count
            denom = np.sqrt(var_x * var_y)
            if denom != 0:
                ar1[i] = cov / denom
    return ar1


def ar1_series_5yr_optimized(array):
    bands_year = 24  # 根据实际情况调整
    yrs = 5
    window_size = bands_year * yrs
    min_periods = int(bands_year / 2 * yrs)

    # 优化后的STL参数调整
    stl = STL(array, period=bands_year, seasonal=bands_year + 1, robust=True)
    res = stl.fit().resid

    return sliding_ar1(res, window_size, min_periods)

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/nadir_ndvi_15d_500/'
    grass_folder = current_dir + '/1_Input/grassCover_250m/'
    tree_folder = current_dir + '/1_Input/treeCover_250m/'
    crop_folder = current_dir + '/1_Input/cropCover_250m/'
    water_folder = current_dir + '/1_Input/waterCover_250m/'
    urbanExp_folder = current_dir + '/1_Input/urban_expansion_frac_500m/'
    # urban_folder = current_dir+'/1_Input/urban_area/'
    filenames = os.listdir(EVI_folder)[1:]
    IDs = []
    for name in filenames:
        id = float(name.split('_')[-1].split('.tif')[0])
        IDs.append(id)
    IDs = np.sort(IDs)
    tacs = []
    for id in IDs:
        EVI0 = tf.imread(EVI_folder + 'nadir_ndvi_15d_' + str(id) + '.tif')[:, :, :264*2]
        nan_frac = np.sum(np.isnan(EVI0),axis=2)/EVI0.shape[2]
        # plt.figure(); plt.hist(nan_frac.reshape(-1),50)
        crop_frc = tf.imread(crop_folder + 'cropC_' + str(id) + '.tif')
        crop_frc = cv2.resize(crop_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
        # plt.figure(); plt.imshow(water_frc)

        grass_frc = tf.imread(grass_folder + 'grassC_' + str(id) + '.tif')
        grass_frc = cv2.resize(grass_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
        tree_frc = tf.imread(tree_folder + 'treeC_' + str(id) + '.tif')
        tree_frc = cv2.resize(tree_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
        water_frc = tf.imread(water_folder + 'waterC_' + str(id) + '.tif')
        water_frc = cv2.resize(water_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
        urbanExp_frc = tf.imread(urbanExp_folder + 'urban_exp_C_' + str(id) + '.tif')

        # remove the pixel with crop > 20% or urban expansion or miss data >30%
        crop_frc[crop_frc > 0.2] = np.nan
        veg_nature = grass_frc + tree_frc
        crop_frc[crop_frc > veg_nature] = np.nan
        mask_miss = np.sum(np.isnan(EVI0), axis=2) / EVI0.shape[2]
        mask = np.isnan(crop_frc) | (mask_miss > 0.3) | (water_frc>0.3) | (urbanExp_frc>0.2)
        # plt.figure(); plt.imshow(mask)
        crop_mask_1d = mask.flatten()
        EVI_rsp = EVI0.reshape((EVI0.shape[0] * EVI0.shape[1], EVI0.shape[2]))
        EVI = EVI_rsp[~crop_mask_1d, :]

        # fill nan with climatology; twice
        yr_num = 22
        bands_year = 24
        EVI = fill_with_climatology(EVI, yr_num, bands_year)
        EVI = fill_with_climatology(EVI, yr_num, bands_year)

        if EVI.shape[0]<10: continue

        # 分块并行处理
        chunk_size = 1000
        chunks = [EVI[i:i + chunk_size] for i in range(0, EVI.shape[0], chunk_size)]

        with mp.Pool(min(os.cpu_count(), 100)) as pool:
            results = []
            for chunk in chunks:
                results.extend(pool.map(ar1_series_5yr_optimized, chunk))

        ar1_res_5sg = np.array(results)

        # plt.figure(); plt.plot(np.linspace(2001, 2023 + 11 / 12, 276), np.nanmean(ar1_res_5sg, axis=0))
        # res_map = EVI_rsp.astype(float) * 1
        # res_map[~np.isnan(crop_mask_1d), :] = res
        # res_map[np.isnan(crop_mask_1d), :] = np.nan
        # res_map = res_map.reshape((EVI0.shape[0], EVI0.shape[1], EVI0.shape[2]))

        tac_map = EVI_rsp.astype(float) * 1
        tac_map[~crop_mask_1d, :] = ar1_res_5sg
        tac_map[crop_mask_1d, :] = np.nan
        tac_map = tac_map.reshape((EVI0.shape[0], EVI0.shape[1], EVI0.shape[2]))
        tac_spa_mean = np.nanmean(tac_map, axis=2)
        output_file = current_dir + '/2_Output/tac_500m/' + 'tac_STL_' + str(id) + '.npy'
        np.save(output_file, tac_map)
        print(id)

