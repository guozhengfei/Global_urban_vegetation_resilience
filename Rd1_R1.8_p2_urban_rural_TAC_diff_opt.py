import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;

matplotlib.use('Qt5Agg')
import scipy.signal as ss
#import multiprocess as mp
import os
import warnings

warnings.filterwarnings("ignore")
import cv2
import time


def ar1_series_5yr(array):
    import numpy.ma as ma
    from sklearn.linear_model import LinearRegression
    def calc_ar1(x):
        return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0, 1]
    X = array[:, :-1]
    y = array[:, -1]
    regressor = LinearRegression()
    regressor.fit(X, y)
    y_rmv = regressor.coef_[1] * X[:, 1] + regressor.coef_[2] * X[:, 2] + regressor.coef_[3] * X[:, 3]
    y_final = y - y_rmv
    ar1 = calc_ar1(y_final)
    return ar1


def IQR_filter2(array):
    p25 = np.nanpercentile(array, 25, axis=1)
    p75 = np.nanpercentile(array, 75, axis=1)
    IQR = p75 - p25
    maxV = np.tile(p75 + 0.5 * IQR, (array.shape[1], 1)).T
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

def stable_landcover_mask(landcover_file, target_shape):
    landcover = tf.imread(landcover_file)
    if landcover.ndim != 3:
        raise ValueError(f'Expected a 3-band land cover tif: {landcover_file}')
    if landcover.shape[0] == 3 and landcover.shape[-1] != 3:
        landcover = np.moveaxis(landcover, 0, -1)
    if landcover.shape[-1] != 3:
        raise ValueError(f'Expected exactly 3 land cover bands: {landcover_file}')

    stable = np.all(landcover == landcover[:, :, [0]], axis=2)
    if stable.shape != target_shape:
        stable = cv2.resize(stable.astype(np.uint8), (target_shape[1], target_shape[0]), cv2.INTER_NEAREST).astype(bool)

    mask = stable.astype(float)
    mask[~stable] = np.nan
    return mask

# New: batch AR1 computation (chunked) to avoid per-pixel process overhead and sklearn fit for each pixel
def ar1_series_batch(arr, chunk_size=16000):
    """
    arr: numpy array shape (n_pixels, T, F) where F>=2 and last column is y, first F-1 columns are X.
    Processes in chunks to limit memory and avoid heavy per-row overhead.
    Returns: 1D array of AR(1) values per pixel (nan where insufficient data).
    """
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=float)

    for start in range(0, n, chunk_size):
        chunk = arr[start:start + chunk_size]  # shape (m, T, F)
        m = chunk.shape[0]
        for i in range(m):
            array = chunk[i]
            # X: all columns except last, y: last column
            X = array[:, :-1]
            y = array[:, -1]
            # drop rows with NaN in X or y
            valid_rows = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            if valid_rows.sum() < 4:
                continue
            # least squares solve for coefficients
            try:
                coef, *_ = np.linalg.lstsq(X[valid_rows], y[valid_rows], rcond=None)
            except Exception:
                continue
            # require at least 4 coefficients for same indexing as original code
            if coef.shape[0] < 4:
                continue
            # reconstruct contribution of selected coef indices (1,2,3) as original code did
            # ensure X has columns for indices used
            if X.shape[1] <= 3:
                continue
            y_rmv = coef[1] * X[:, 1] + coef[2] * X[:, 2] + coef[3] * X[:, 3]
            y_final = y - y_rmv
            v1 = y_final[:-1]
            v2 = y_final[1:]
            valid_ar = ~(np.isnan(v1) | np.isnan(v2))
            if valid_ar.sum() < 2:
                continue
            out[start + i] = np.corrcoef(v1[valid_ar], v2[valid_ar])[0, 1]
    return out


if __name__ == '__main__':
    t0 = time.time()
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/nadir_ndvi_15d_500/'
    grass_folder = current_dir + '/1_Input/grassCover_250m/'
    tree_folder = current_dir + '/1_Input/treeCover_250m/'
    crop_folder = current_dir + '/1_Input/cropCover_250m/'
    water_folder = current_dir + '/1_Input/waterCover_250m/'
    urbanExp_folder = current_dir + '/1_Input/urban_expansion_frac_500m/'
    filenames = os.listdir(current_dir + '/1_Input/ta_anom/')
    disturbance_folder = current_dir + '/2_Output/Modis_recovery_resistance/'
    urban_folder_2018 = current_dir + '/1_Input/urban/urban_2018_1000m/'
    urban_folder_1990 = current_dir + '/1_Input/urban/urban_1990_250m/'
    landcover_folder = current_dir + '/1_Input/landcover_modis_500m/'



    IDs = []
    for name in filenames:
        id = float(name.split('_')[-1].split('.npy')[0])
        IDs.append(id)

    TAC = []
    IDs_num = []
    for id in IDs[1:]:
        EVI0 = tf.imread(EVI_folder + 'nadir_ndvi_15d_' + str(id) + '.tif')[:, :, :264 * 2][:, :, ::2]
        nan_frac = np.sum(np.isnan(EVI0), axis=2) / EVI0.shape[2]
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
        landcover_stable = stable_landcover_mask(landcover_folder + 'landcover_500m_' + str(id) + '.tif', EVI0.shape[:2])

        mask = np.isnan(crop_frc) | (mask_miss > 0.3) | (water_frc > 0.3) | (urbanExp_frc > 0.2) | (np.isnan(landcover_stable))
        # plt.figure(); plt.imshow(mask)
        crop_mask_1d = mask.flatten()
        EVI_rsp = EVI0.reshape((EVI0.shape[0] * EVI0.shape[1], EVI0.shape[2]))

        EVI = EVI_rsp[~crop_mask_1d, :]
        EVI = IQR_filter2(EVI)

        # Fill NaN values using 5-year climatology
        EVI = fill_nan_with_climatology(EVI)

        ser = EVI
        if ser.shape[0] < 10: continue
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
        # res_ndvi[np.isnan(res_ndvi)] = 0
        # distur = np.load(disturbance_folder + 'extreme_events_' + str(id) + '_smith_' + '2' + 'sd.npy')
        # res_ndvi[distur] = np.nan

        nan_frac_ndvi = np.sum(np.isnan(res_ndvi), axis=1) / res_ndvi.shape[1]
        res_pr_org = np.load(current_dir + '/1_Input/pre_anom/pr_month_anomaly_' + str(id) + '.npy')[:, :, 1:1 + 264]
        res_pr = res_pr_org.reshape((res_pr_org.shape[0] * res_pr_org.shape[1], res_pr_org.shape[2]))[~crop_mask_1d, :]
        nan_frac_pr = np.sum(np.isnan(res_pr), axis=1) / res_pr.shape[1]

        res_rad_org = np.load(current_dir + '/1_Input/rad_anom/rad_month_anomaly_' + str(id) + '.npy')[:, :, 1:1 + 264]
        res_rad = res_rad_org.reshape((res_rad_org.shape[0] * res_rad_org.shape[1], res_rad_org.shape[2]))[
                  ~crop_mask_1d, :]
        nan_frac_rad = np.sum(np.isnan(res_rad), axis=1) / res_rad.shape[1]

        res_ta_org = np.load(current_dir + '/1_Input/ta_anom/ta_month_anomaly_' + str(id) + '.npy')[:, :, 1:1 + 264]
        res_ta = res_ta_org.reshape((res_ta_org.shape[0] * res_ta_org.shape[1], res_ta_org.shape[2]))[~crop_mask_1d, :]
        nan_frac_ta = np.sum(np.isnan(res_ta), axis=1) / res_ta.shape[1]

        # mask = (nan_frac_ndvi > 0.75) | (nan_frac_pr > 0.3) | (nan_frac_rad > 0.3) | (nan_frac_ta > 0.3)

        variables_all = np.stack((res_ndvi[:, :-1], res_ta[:, 1:], res_rad[:, 1:], res_pr[:, 1:], res_ndvi[:, 1:]), axis=2)

        # variables_all = variables_all[~mask, :, :]
        del rm_offline, Evi_sea_rep

        # Use batch AR1 computation instead of per-pixel Pool.map
        tac = ar1_series_batch(variables_all, chunk_size=16000)
        tac[tac<0]=np.nan

        urban_2018 = tf.imread(urban_folder_2018 + 'fvc_' + str(id) + '.tif').astype(float)
        urban_2018_rsz = cv2.resize(urban_2018, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_NEAREST)
        urban_2018_rsz[urban_2018_rsz != float(id)] = np.nan
        urban_1990 = tf.imread(urban_folder_1990 + 'urban_1990_' + str(id) + '.tif').astype(float)
        urban_1990= cv2.resize(urban_1990, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_NEAREST)
        urban_1990[urban_1990 == 0] = np.nan
        urban_1990_r = urban_1990.reshape(-1)[~crop_mask_1d]

        rural_near = urban_2018_rsz * 1  # rural-urban interface
        for i in range(3*2):
            rural_near = extend_edge(rural_near)

        rural_bgr = rural_near * 1  # rural background
        for i in range(10*2):
            rural_bgr = extend_edge(rural_bgr)

        # URBAN 1990

        tac_urban_1990 = np.nanmean(tac+urban_1990_r-urban_1990_r)

        # URBAN 1990-2018
        urban_1990_v2 = urban_1990 * 1
        urban_1990_v2[np.isnan(urban_1990_v2)] = 0
        urban_1990_v2[urban_1990_v2 != 0] = np.nan
        urban_1990_2018 = rural_near + urban_1990_v2
        # plt.figure(); plt.imshow(urban_1990_2018)
        urban_1990_2018 = urban_1990_2018.reshape(-1)[~crop_mask_1d]

        tac_urban_1990_2018 = np.nanmean(tac+urban_1990_2018-urban_1990_2018)
        # plt.plot(np.linspace(2004, 2021 , 18), tac_urban_1990_2018)

        # RURAL-BACKGROUND
        rural_near_v2 = rural_near * 1
        rural_near_v2[np.isnan(rural_near_v2)] = 0
        rural_near_v2[rural_near_v2 != 0] = np.nan
        urban_rural_bgr = rural_bgr + rural_near_v2
        urban_rural_bgr = urban_rural_bgr.reshape(-1)[~crop_mask_1d]

        tac_rural_bgr = np.nanmean(tac+urban_rural_bgr-urban_rural_bgr)
        # plt.plot(np.linspace(2004, 2021, 18),tac_rural_bgr)
        IDs_num.append(id)

        TAC.append([tac_urban_1990, tac_urban_1990_2018, tac_rural_bgr])
        print(id)
    TAC_arr = np.array(TAC)
    IDs_num = np.array(IDs_num)

    output_file = current_dir + '/2_Output/' + 'tac_nadir_city_3zones_no_lcc_modis' + '.npz'
    np.savez(output_file, array1=TAC_arr, array2=IDs_num)
    print("Done in %.1f s" % (time.time() - t0))
