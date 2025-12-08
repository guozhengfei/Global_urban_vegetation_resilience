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
    CORE_SD_LEVELS = [1, 1.5, 2, 2.5]  # Disturbance intensity levels
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
        gs_mask = np.tile(vi_seasonal > vi_threshold, YR_NUM)

        # --- De-seasonalize ---
        seasonal_mean = np.mean(vis_monthly, axis=1)
        seasonal_mean_tiled = np.tile(seasonal_mean, (1, YR_NUM))
        deseasonalized = vis - seasonal_mean_tiled

        # --- De-trend (Rolling Mean) ---
        # Convert to DataFrame to leverage optimized rolling function
        # Transpose to (Time, Pixels) because rolling works on index
        df_deseas = pd.DataFrame(deseasonalized.T)

        # Calculate Rolling Mean (Trend)
        # center=True ensures the trend is aligned with the event
        # min_periods=1 ensures edges are handled
        ROLLING_WINDOW = 12*7
        rolling_trend = df_deseas.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()

        # Calculate Residuals (Anomaly - Trend)
        residuals = deseasonalized - rolling_trend.values.T

        # Apply GS Mask
        residuals[:, ~gs_mask] = 0
        residuals[np.isnan(residuals)] = 0

        # --- Smoothing ---
        gsm_residuals = ss.savgol_filter(residuals, 18, 1, mode='nearest', axis=1)
        gsm_residuals = ss.savgol_filter(gsm_residuals, 7, 1, mode='nearest', axis=1)

        # Add Static Mean back to standardize the baseline for thresholding
        vis_mean_pixel = np.nanmean(vis, axis=1, keepdims=True)
        gsm_arr = gsm_residuals + vis_mean_pixel
        # --- Statistics ---
        mean_all = np.nanmean(gsm_arr, axis=1)
        std_all = np.nanstd(gsm_arr, axis=1)
        n_pixels, n_times = gsm_arr.shape

        # --- Loop through n = 1, 2, 3 ---
        for n_sd in CORE_SD_LEVELS:

            # Define Thresholds
            boundary_thresholds = mean_all - (BOUNDARY_SD * std_all)
            core_thresholds = mean_all - (n_sd * std_all)

            # Create boundary mask
            boundary_mask = gsm_arr <= boundary_thresholds[:, np.newaxis]

            # Output Arrays
            resistance_out = np.full(n_pixels, np.nan)
            resilience_out = np.full(n_pixels, np.nan)
            dVI_out = np.full(n_pixels, np.nan)
            extreme_events_out = np.zeros((n_pixels, n_times), dtype=bool)

            total_events = 0

            for i in range(n_pixels):
                pixel_bool = boundary_mask[i, :]
                if not np.any(pixel_bool): continue

                # Find contiguous periods
                padded = np.concatenate(([False], pixel_bool, [False]))
                diffs = np.diff(padded.astype(int))
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0] - 1  # inclusive

                res_list = []
                rec_list = []
                dVI_list = []

                for start, end in zip(starts, ends):
                    duration = end - start + 1
                    if duration < 4: continue

                    pixel_data = gsm_arr[i, :]
                    event_segment = pixel_data[start: end + 1]

                    # --- CRITICAL CHECK: Depth > n SD ---
                    min_val = np.min(event_segment)
                    if min_val > core_thresholds[i]:
                        continue

                        # Valid Event
                    total_events += 1
                    extreme_events_out[i, start: end + 1] = True

                    # Calculate Metrics
                    min_idx_local = np.argmin(event_segment)
                    min_idx_global = start + min_idx_local

                    # Resistance (Slope to Min)
                    resist_slice = pixel_data[start: min_idx_global + 1]
                    res_list.append(get_slope(resist_slice))

                    # Resilience (Slope from Min)
                    rec_slice = pixel_data[min_idx_global: end + 1]
                    rec_list.append(get_slope(rec_slice))

                    # dT (max to Min)
                    dVI_slice = pixel_data[min_idx_global] - pixel_data[start]
                    dVI_list.append(dVI_slice)

                if res_list: resistance_out[i] = np.nanmean(res_list)
                if rec_list: resilience_out[i] = np.nanmean(rec_list)
                if dVI_list: dVI_out[i] = np.nanmean(dVI_list)

            # Calculate average events
            avg_events = total_events / n_pixels

            # Save
            suffix = f"{id}_smith_{n_sd}sd.npy"
            np.save(f"{output_folder}resistance_{suffix}", resistance_out)
            np.save(f"{output_folder}resilience_{suffix}", resilience_out)
            np.save(f"{output_folder}dVI_{suffix}", dVI_out)
            np.save(f"{output_folder}extreme_events_{suffix}", extreme_events_out)

            print(
                f"  -> n={n_sd}SD: {np.sum(~np.isnan(resistance_out))} pixels affected. Avg events/pixel: {avg_events:.4f}")


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
            output_file_urbanlabel = current_dir + '/2_Output/Modis_recovery_resistance/' + 'urban_label_' + str(
                id) + '_.npy'

            np.save(output_file_urbanlabel, urban_label2)
