import numpy as np
import tifffile as tf
# import matplotlib.pyplot as plt
# import matplotlib; matplotlib.use('Qt5Agg')
import scipy.signal as ss
import multiprocess as mp
import os
import warnings

warnings.filterwarnings("ignore")
import cv2


def ar1_series_5yr(array):
    yrs = 5  # 3,5,7
    import pandas as pd
    import numpy.ma as ma

    def calc_ar1(x):
        return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0, 1]

    bands_year = 12

    t = bands_year * yrs
    ar1 = pd.Series(array).rolling(t, min_periods=6 * yrs, center=True).apply(calc_ar1).values
    return ar1


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/org_ndvi_monthly_500/'
    grass_folder = current_dir + '/1_Input/grassCover_250m/'
    tree_folder = current_dir + '/1_Input/treeCover_250m/'
    crop_folder = current_dir + '/1_Input/cropCover_250m/'
    # urban_folder = current_dir+'/1_Input/urban_area/'
    filenames = os.listdir(EVI_folder)[1:]

    IDs = []
    for name in filenames:
        id = name.split('_')[-1].split('.tif')[0]
        IDs.append(id)

    tacs = []
    for id in IDs:
        EVI0 = tf.imread(EVI_folder + 'ndvi_monthly_' + id + '.tif')[:, :, :264]
        crop_frc = tf.imread(crop_folder + 'cropC_' + id + '.tif')
        crop_frc = cv2.resize(crop_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)

        grass_frc = tf.imread(grass_folder + 'grassC_' + id + '.tif')
        grass_frc = cv2.resize(grass_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)

        tree_frc = tf.imread(tree_folder + 'treeC_' + id + '.tif')
        tree_frc = cv2.resize(tree_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
        crop_frc[crop_frc > 0.2] = np.nan
        veg_nature = grass_frc + tree_frc
        crop_frc[crop_frc > veg_nature] = np.nan
        mask_miss = np.sum(np.isnan(EVI0), axis=2) / EVI0.shape[2]
        mask = np.isnan(crop_frc) | (mask_miss > 0.5)
        # plt.figure(); plt.imshow(mask)
        crop_mask_1d = mask.flatten()
        EVI_rsp = EVI0.reshape((EVI0.shape[0] * EVI0.shape[1], EVI0.shape[2]))
        EVI = EVI_rsp[~crop_mask_1d, :]
        EVI_sg = ss.savgol_filter(EVI.T, 4, 3, mode='nearest', axis=0)
        EVI_sg[EVI_sg < 0] = 0

        # plt.figure(); plt.plot(np.linspace(2001,2021+11/12,264),np.nanmean(EVI_sg, axis=1))
        ser = EVI_sg.T
        EVI_yr = np.zeros_like(ser)

        yr_num = 22
        bands_year = 12
        for year in range(yr_num):
            st = year * bands_year
            ed = st + bands_year
            evi_year = np.nanmean(ser[:, st:ed], axis=1)
            evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
            EVI_yr[:, st:ed] = evi_year_rep
        rm_offline = ser - EVI_yr
        # plt.figure(); plt.plot(np.nanmean(rm_offline, axis=0))

        del EVI_yr, ser

        Evi_sea_rep = np.zeros_like(rm_offline)
        for yr in range(yr_num):
            start_index = (yr - 4) * bands_year
            if start_index < 0: start_index = 0
            end_index = (yr + 5) * bands_year
            if end_index > bands_year * yr_num: end_index = bands_year * yr_num
            data_i = rm_offline[:, start_index:end_index]
            Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)),
                              axis=1)
            Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea

        res = rm_offline - Evi_sea_rep
        res[np.isnan(res)] = 0

        divided_arrays = [row for row in res]
        with mp.Pool(240) as pool:
            results = list(pool.map(ar1_series_5yr, divided_arrays))
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

        # plt.figure(); plt.imshow(tac_spa_mean,cmap='RdYlBu_r',vmin=0.3,vmax=0.6)
        # urban_mask[urban_mask!=float(id)]=np.nan
        # plt.imshow(urban_mask, alpha=0.3)
        # plt.figure(); plt.hist(tac_spa_mean[urban_mask == float(id)], 50, range=[0.2, 0.6]);
        # plt.figure(); plt.hist(tac_spa_mean[urban_mask != float(id)], 50, range=[0.2, 0.6]);
        # plt.figure(); plt.plot(np.linspace(2001, 2023 + 11 / 12, 276), np.nanmean(ar1_res_5sg, axis=0))
        # plt.plot(np.linspace(2001, 2023 + 11 / 12, 276), np.nanmean(ar1_res_5sg[urban_mask_2 == 16, :], axis=0))
        # plt.plot(np.linspace(2001, 2023 + 11 / 12, 276), np.nanmean(ar1_res_5sg[urban_mask_2 != 16, :], axis=0))

        # plt.figure();
        # plt.plot(np.nanmean(EVI, axis=1), np.nanmean(ar1_res_5sg, axis=1), '.')

        output_file = current_dir + '/2_Output/tac_500_noBRDF/' + 'tac_' + id + '.npy'
        np.save(output_file, tac_map)

        print(id)

