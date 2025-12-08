import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.signal as ss
import multiprocess as mp
import os
import warnings

warnings.filterwarnings("ignore")
import cv2

def ar1_series_5yr(array):
    yrs = 4  # 3,5,7
    import pandas as pd
    import numpy.ma as ma
    from sklearn.linear_model import LinearRegression

    def calc_ar1(x):
        return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0, 1]

    X = array[:,:-1]
    y = array[:,-1]
    regressor = LinearRegression()
    regressor.fit(X, y)
    y_rmv = regressor.coef_[1] * X[:,1] + regressor.coef_[2] * X[:,2] + regressor.coef_[3] * X[:,3]
    y_final = y - y_rmv
    bands_year = 12
    t = bands_year*yrs
    ar1 = pd.Series(y_final).rolling(t, center=True).apply(calc_ar1).values
    return ar1

def IQR_filter2(array):
    p25 = np.nanpercentile(array, 25,axis=1)
    p75 = np.nanpercentile(array,75,axis=1)
    IQR = p75-p25
    maxV = np.tile(p75 + 0.5*IQR, (array.shape[1], 1)).T
    minV = np.tile(p25 - 0.5 * IQR, (array.shape[1], 1)).T
    array[array < minV] = np.nan
    array[array > maxV] = np.nan
    arraynew = array
    return arraynew

# Add the fill_nan_with_climatology function
def fill_nan_with_climatology(EVI, bands_year=12):
    num_years = EVI.shape[1] // bands_year
    climatology = np.zeros((EVI.shape[0], EVI.shape[1]))
    
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
            EVI[:, start:end].reshape(EVI.shape[0], 5, bands_year), axis=1
        )
    
    EVI_filled = np.where(np.isnan(EVI), climatology, EVI)
    
    return EVI_filled

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/nadir_ndvi_15d_500/'
    grass_folder = current_dir + '/1_Input/grassCover_250m/'
    tree_folder = current_dir + '/1_Input/treeCover_250m/'
    crop_folder = current_dir + '/1_Input/cropCover_250m/'
    water_folder = current_dir + '/1_Input/waterCover_250m/'
    urbanExp_folder = current_dir + '/1_Input/urban_expansion_frac_500m/'
    # urban_folder = current_dir+'/1_Input/urban_area/'
    filenames = os.listdir(current_dir + '/1_Input/ta_anom/')

    IDs = []
    for name in filenames:
        id = float(name.split('_')[-1].split('.npy')[0])
        IDs.append(id)

    IDs = np.sort(IDs)
    IDs_2 = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array2']
    IDs_diff = set(IDs) - set(IDs_2)

    tacs = []
    for id in IDs_diff:#IDs:
        EVI0 = tf.imread(EVI_folder + 'nadir_ndvi_15d_' + str(id) + '.tif')[:, :, :264*2][:,:,::2]
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
        # plt.figure(); plt.plot(np.linspace(2001,2021+11/12,264),EVI[1,:])
        # plt.figure(); plt.plot(np.linspace(2001,2021+11/12,264),EVI_sg.T[1,:])

        EVI = EVI_rsp[~crop_mask_1d, :]
        EVI = IQR_filter2(EVI)
        
        # Fill NaN values using 5-year climatology
        EVI = fill_nan_with_climatology(EVI)
        
        # EVI_sg = ss.savgol_filter(EVI.T, 4, 3, mode='nearest', axis=0)
        # EVI_sg[EVI_sg < 0] = 0
        # ser = EVI_sg.T
        ser = EVI
        if ser.shape[0]<10: continue
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

        # res = rm_offline - Evi_sea_rep
        res_ndvi = rm_offline - Evi_sea_rep
        # plt.figure(); plt.plot(np.nanmean(res_pr, axis=0))
        res_ndvi[np.isnan(res_ndvi)]=0

        nan_frac_ndvi = np.sum(np.isnan(res_ndvi), axis=1) / res_ndvi.shape[1]
        res_pr_org = np.load(current_dir + '/1_Input/pre_anom/pr_month_anomaly_'+str(id)+'.npy')[:, :,1:1+264]
        res_pr = res_pr_org.reshape((res_pr_org.shape[0]*res_pr_org.shape[1],res_pr_org.shape[2]))[~crop_mask_1d, :]
        nan_frac_pr = np.sum(np.isnan(res_pr), axis=1) / res_pr.shape[1]

        res_rad_org = np.load(current_dir + '/1_Input/rad_anom/rad_month_anomaly_' + str(id) + '.npy')[:, :, 1:1 + 264]
        res_rad = res_rad_org.reshape((res_rad_org.shape[0] * res_rad_org.shape[1], res_rad_org.shape[2]))[~crop_mask_1d, :]
        nan_frac_rad = np.sum(np.isnan(res_rad), axis=1) / res_rad.shape[1]

        res_ta_org = np.load(current_dir + '/1_Input/ta_anom/ta_month_anomaly_' + str(id) + '.npy')[:, :, 1:1 + 264]
        res_ta = res_ta_org.reshape((res_ta_org.shape[0] * res_ta_org.shape[1], res_ta_org.shape[2]))[~crop_mask_1d, :]
        nan_frac_ta = np.sum(np.isnan(res_ta), axis=1) / res_ta.shape[1]

        mask = (nan_frac_ndvi > 0.75) | (nan_frac_pr > 0.3) | (nan_frac_rad > 0.3) | (nan_frac_ta > 0.3)

        variables_all = np.stack((res_ndvi[:, :-1], res_ta[:, 1:], res_rad[:, 1:], res_pr[:, 1:], res_ndvi[:, 1:]), axis=2)

        variables_all = variables_all[~mask, :, :]
        del rm_offline, Evi_sea_rep
        # variables_all = variables_all[::100,:,:]
        divided_arrays = [row for row in variables_all]

        with mp.Pool(100) as pool:
            results = list(pool.map(ar1_series_5yr, divided_arrays))
        ar1_res_5sg = np.array(results)

        # plt.figure(); plt.plot(np.linspace(2001, 2023 + 11 / 12, 276), np.nanmean(ar1_res_5sg, axis=0))
        # res_map = EVI_rsp.astype(float) * 1
        # res_map[~np.isnan(crop_mask_1d), :] = res
        # res_map[np.isnan(crop_mask_1d), :] = np.nan
        # res_map = res_map.reshape((EVI0.shape[0], EVI0.shape[1], EVI0.shape[2]))

        tac_map = EVI_rsp.astype(float)[:,:-1] * 1
        mask2 = crop_mask_1d
        mask2[~mask2][mask] = False
        tac_map[~mask2, :] = ar1_res_5sg
        tac_map[mask2, :] = np.nan
        tac_map = tac_map.reshape((EVI0.shape[0], EVI0.shape[1], EVI0.shape[2]-1))
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

        output_file = current_dir + '/2_Output/tac_500m/' + 'tac_' + str(id) + '_v2.npy'
        np.save(output_file, tac_map)
        print(id)

