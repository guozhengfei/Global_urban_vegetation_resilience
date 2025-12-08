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
from sklearn.linear_model import LinearRegression

# Refined function to calculate rolling variance with improved efficiency
def vectorized_processing(variables_all):
    """
    Calculates rolling variance on residuals after a vectorized linear regression.
    This function is optimized to process a 3D array of (pixels, time, features)
    without looping through pixels, providing a significant performance boost.
    """
    yrs = 4
    bands_year = 12
    t_window = bands_year * yrs

    def rolling_var_centered_vectorized(a, window):
        """A fast, centered rolling variance for a 2D array using stride tricks."""
        half_window = window // 2
        # Pad with NaNs to handle boundaries for centered window calculation
        padded_a = np.pad(a.astype(float), ((0, 0), (half_window, half_window)), mode='constant', constant_values=np.nan)
        
        # Create a view of rolling windows without copying data
        shape = (a.shape[0], a.shape[1], window)
        strides = (padded_a.strides[0], padded_a.strides[1], padded_a.strides[1])
        rolling_a = np.lib.stride_tricks.as_strided(padded_a, shape=shape, strides=strides, writeable=False)
        
        # Calculate variance over each window, ignoring NaNs
        return np.nanvar(rolling_a, axis=2)

    X = variables_all[:, :, :-1]
    y = variables_all[:, :, -1]

    # Add a constant for the intercept term to the features
    X_with_const = np.concatenate([X, np.ones((*X.shape[:-1], 1))], axis=-1)

    # Solve for regression coefficients for all pixels at once using np.linalg.lstsq
    try:
        coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        # Fallback for robustness in case of singular matrices
        print("Warning: np.linalg.lstsq failed. Falling back to a slower, iterative approach.")
        coeffs = np.full((X.shape[0], X_with_const.shape[2]), np.nan)
        for i in range(X.shape[0]):
            try:
                coeffs[i,:] = np.linalg.lstsq(X_with_const[i,:,:], y[i,:], rcond=None)[0]
            except np.linalg.LinAlgError:
                continue  # Keep coefficients as NaN for this pixel

    # Vectorized custom residual calculation based on the original logic
    # y_rmv = c1*X1 + c2*X2 + c3*X3 (where X1, X2, X3 are ta, rad, pr)
    y_rmv = np.sum(X[:, :, 1:4] * coeffs[:, np.newaxis, 1:4], axis=2)
    y_final = y - y_rmv  # Residuals

    # Calculate rolling variance on the residuals for all pixels
    variance = rolling_var_centered_vectorized(y_final, t_window)
    
    return variance

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

def IQR_filter(array):
    mean_V = np.nanmean(array, axis=1)
    sd_V = np.nanstd(array, axis=1)

    maxV = np.tile(mean_V + 3*sd_V, (array.shape[1], 1)).T
    minV = np.tile(mean_V - 3*sd_V, (array.shape[1], 1)).T
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
    Data_folder = os.path.dirname(current_dir) + '/urban_env_data/Ta_ear5Land_urban/'


    IDs = []
    for name in filenames:
        id = float(name.split('_')[-1].split('.npy')[0])
        IDs.append(id)

    IDs = np.sort(IDs)
    IDs_2 = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array2']
    IDs_diff = set(IDs) - set(IDs_2)

    tacs = []
    for id in IDs:
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
        # plt.figure(); plt.plot(np.nanmean(EVI,axis=0))
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
            start_index = (yr - 3) * bands_year
            if start_index < 0: start_index = 0
            end_index = (yr + 4) * bands_year
            if end_index > bands_year * yr_num: end_index = bands_year * yr_num
            data_i = rm_offline[:, start_index:end_index]
            Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)),
                              axis=1)
            Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea

        # res = rm_offline - Evi_sea_rep
        res_ndvi = rm_offline - Evi_sea_rep

        # only focus on growing season
        ta_i = tf.imread(Data_folder + 'Ta_monthly_' + str(id) + '.tif')  # [:, :, :264 * 2]
        ta_i = np.nanmedian(np.nanmedian(ta_i, axis=0), axis=0)
        ta_i_rsp = np.reshape(ta_i, (int(ta_i.shape[0] / 12), 12))
        ta_i_sea = np.nanmedian(ta_i_rsp, axis=0) - 273.15
        gs_mask = ta_i_sea > 5 # monthly mean max Tair larger than 5 degree
        gs_mask = np.repeat(gs_mask[np.newaxis], yr_num, axis=0).reshape(-1)
        res_ndvi[:,~gs_mask]=np.nan

        # plt.figure(); plt.plot(np.nanmean(res_ndvi, axis=0))
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

        # Process the entire data matrix at once using vectorized operations
        print(f"Processing {variables_all.shape[0]} pixels...")
        ar1_res_5sg = vectorized_processing(variables_all)

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
        
        # Calculate long-term EVI mean for each pixel
        EVI0_mean = np.nanmean(EVI0, axis=2)
        EVI0_mean[EVI0_mean == 0] = np.nan  # Avoid division by zero
        tac_map = tac_map**0.5 / EVI0_mean[:, :, np.newaxis]
        
        tac_spa_mean = np.nanmean(tac_map, axis=2)

        # Calculate and save yearly mean TAC
        print(f"Calculating yearly mean for {id}...")
        n_rows, n_cols, n_months = tac_map.shape
        n_years = n_months // 12
        n_rem_months = n_months % 12

        # Process full years
        if n_years > 0:
            full_years_data = tac_map[:, :, :n_years * 12].reshape(n_rows, n_cols, n_years, 12)
            yearly_means = np.nanmean(full_years_data, axis=3)
        else:
            yearly_means = np.empty((n_rows, n_cols, 0))

        # Process remaining months if any
        if n_rem_months > 0:
            rem_data = tac_map[:, :, n_years * 12:]
            rem_mean = np.nanmean(rem_data, axis=2, keepdims=True)
            if n_years > 0:
                final_yearly_mean = np.concatenate((yearly_means, rem_mean), axis=2)
            else:
                final_yearly_mean = rem_mean
        else:
            final_yearly_mean = yearly_means

        # output_file_yearly = current_dir + '/2_Output/var_500_yr/' + 'var_yearly_mean_' + str(id) + '.npy'
        output_file_yearly = current_dir + '/2_Output/var_500_yr/' + 'cv_yearly_mean_' + str(id) + '.npy'

        np.save(output_file_yearly, final_yearly_mean)
        print(f"Yearly mean TAC saved for {id}")

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

        # output_file = current_dir + '/2_Output/tac_500m/' + 'tac_' + str(id) + '_variance.npy'
        # np.save(output_file, tac_map)
        # print(id)

