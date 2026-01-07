import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib;matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.signal as ss
import warnings
warnings.filterwarnings("ignore")
from scipy.ndimage import convolve1d
import os
import multiprocess as mp
import cv2

def get_slope(y):
    """Calculates linear slope using polyfit."""
    if len(y) < 2: return np.nan
    return np.polyfit(np.arange(len(y)), y, 1)[0]


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


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/nadir_ndvi_15d_500/'
    grass_folder = current_dir + '/1_Input/grassCover_250m/'
    tree_folder = current_dir + '/1_Input/treeCover_250m/'
    crop_folder = current_dir + '/1_Input/cropCover_250m/'
    water_folder = current_dir + '/1_Input/waterCover_250m/'
    urbanExp_folder = current_dir + '/1_Input/urban_expansion_frac_500m/'
    urban_folder_2018 = current_dir + '/1_Input/urban/urban_2018_1000m/'
    urban_folder_1990 = current_dir + '/1_Input/urban/urban_1990_250m/'
    filenames = os.listdir(current_dir + '/1_Input/ta_anom/')

    output_folder = current_dir + '/2_Output/Modis_recovery_resistance/'


    IDs = []
    for name in filenames:
        id = float(name.split('_')[-1].split('.npy')[0])
        IDs.append(id)

    IDs = np.sort(IDs)

    even_nums = []
    BOUNDARY_SD = 0.5  # Start/End definition
    CORE_SD_LEVELS = [2]  # Disturbance intensity levels
    for id in IDs[1:]:
        EVI0 = tf.imread(EVI_folder + 'nadir_ndvi_15d_' + str(id) + '.tif')[:, :, :264 * 2][:, :, ::2]
        nan_frac = np.sum(np.isnan(EVI0), axis=2) / EVI0.shape[2]
        crop_frc = tf.imread(crop_folder + 'cropC_' + str(id) + '.tif')
        crop_frc = cv2.resize(crop_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
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
        crop_frc[grass_frc < tree_frc] = np.nan
        crop_frc[grass_frc < 0.3] = np.nan
        mask_miss = np.sum(np.isnan(EVI0), axis=2) / EVI0.shape[2]
        mask = np.isnan(crop_frc) | (mask_miss > 0.3) | (water_frc > 0.3) | (urbanExp_frc > 0.2)
        crop_mask_1d = mask.flatten()
        EVI_rsp = EVI0.reshape((EVI0.shape[0] * EVI0.shape[1], EVI0.shape[2]))

        EVI = EVI_rsp[~crop_mask_1d, :]
        EVI = IQR_filter2(EVI)

        # Fill NaN values using 5-year climatology
        vis = fill_nan_with_climatology(EVI)

        if vis.shape[0] < 10: continue

        YR_NUM = 22
        BANDS_YEAR = 12

        # Growing Season Mask
        vis_monthly = vis.reshape(vis.shape[0], YR_NUM, BANDS_YEAR)
        vi_seasonal = np.nanmedian(np.nanmedian(vis_monthly, axis=1), axis=0)
        vi_threshold = min(np.nanpercentile(vi_seasonal, 20), 0.2)
        gs_mask = (vi_seasonal > vi_threshold)
        vis_monthly[:,:,~gs_mask]=np.nan
        vis_yearly = np.nanmean(vis_monthly,axis=2)

        # --- De-trend (Rolling Mean) ---

        df_deseas = pd.DataFrame(vis_yearly.T)

        # Calculate Rolling Mean (Trend)
        # center=True ensures the trend is aligned with the event
        # min_periods=1 ensures edges are handled
        ROLLING_WINDOW = 7
        rolling_trend = df_deseas.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()

        # Calculate Residuals (Anomaly - Trend)
        residuals = vis_yearly - rolling_trend.values.T

        # Add Static Mean back to standardize the baseline for thresholding
        vis_mean_pixel = np.nanmean(vis, axis=1, keepdims=True)
        vis_mean_pixel = np.tile(vis_mean_pixel,(YR_NUM))
        gsm_arr = residuals# + vis_mean_pixel
        # --- Statistics ---
        mean_all = np.nanmean(gsm_arr, axis=1)
        std_all = np.nanstd(gsm_arr, axis=1)
        n_pixels, n_times = gsm_arr.shape

        # --- Loop through n = 1, 2, 3 ---
        for n_sd in CORE_SD_LEVELS:
            # Define Thresholds

            core_thresholds = mean_all - (n_sd * std_all)
            core_thresholds_2d = np.tile(core_thresholds,(YR_NUM,1)).T

            # Output Arrays
            dVI_out = gsm_arr+np.nan
            extreme_events_out = np.zeros((n_pixels, n_times), dtype=bool)

            disturbance_label = residuals<core_thresholds_2d
            dVI_out[disturbance_label] = residuals[disturbance_label]/vis_mean_pixel[disturbance_label]
            dVI_out = np.nanmean(dVI_out,axis=1)

            # Calculate average events
            total_events = np.sum(disturbance_label)
            avg_events = total_events / n_pixels
            extreme_events_out[disturbance_label]=True
            extreme_events_out_month = np.repeat(extreme_events_out,12,axis=1)

            # Save
            suffix = f"{id}_smith_{n_sd}sd_grass.npy"

            np.save(f"{output_folder}dVI_{suffix}", dVI_out)
            np.save(f"{output_folder}extreme_events_{suffix}", extreme_events_out_month)

            print(
                f"  {id} -> n={n_sd}SD: {np.sum(~np.isnan(dVI_out))} pixels affected. Avg events/pixel: {avg_events:.4f}")


            urban_2018 = tf.imread(urban_folder_2018 + 'fvc_' + str(id) + '.tif').astype(float)
            urban_2018_rsz = cv2.resize(urban_2018, (EVI0.shape[1], EVI0.shape[0]),cv2.INTER_NEAREST)
            urban_2018_rsz[urban_2018_rsz != float(id)] = np.nan
            urban_1990 = tf.imread(urban_folder_1990 + 'urban_1990_' + str(id) + '.tif').astype(float)
            urban_1990= cv2.resize(urban_1990, (EVI0.shape[1], EVI0.shape[0]),cv2.INTER_NEAREST)
            urban_1990[urban_1990 == 0] = np.nan

            rural_near = urban_2018_rsz * 1  # rural-urban interface
            for i in range(3*2):
                rural_near = extend_edge(rural_near)

            rural_bgr = rural_near * 1  # rural background
            for i in range(10*2):
                rural_bgr = extend_edge(rural_bgr)

            urban_label = urban_1990+np.nan
            urban_label[~np.isnan(rural_bgr)]=0
            urban_label[~np.isnan(rural_near)] = 1
            urban_label[~np.isnan(urban_1990)] = 2
            urban_label2 = urban_label[~mask]
            output_file_urbanlabel = current_dir + '/2_Output/Modis_recovery_resistance/' + 'urban_label_' + str(id) + '_grass.npy'

            np.save(output_file_urbanlabel, urban_label2)
