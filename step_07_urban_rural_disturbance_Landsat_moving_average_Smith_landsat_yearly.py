import numpy as np
import pandas as pd
import scipy.signal as ss
import warnings
import os
import matplotlib;matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def fill_nan_with_climatology(vis, bands_year=12):
    """Vectorized filling of NaNs using seasonal means."""
    n_pixels, n_times = vis.shape
    n_years = n_times // bands_year

    vis_reshaped = vis.reshape(n_pixels, n_years, bands_year)
    seasonality = np.nanmean(vis_reshaped, axis=1)

    seasonality_tiled = np.tile(seasonality, (1, n_years))
    mask = np.isnan(vis)
    vis[mask] = seasonality_tiled[mask]
    return vis


def get_slope(y):
    """Calculates linear slope using polyfit."""
    if len(y) < 2: return np.nan
    return np.polyfit(np.arange(len(y)), y, 1)[0]


if __name__ == '__main__':
    # 1. Path Setup
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    input_folder = current_dir + '/2_Output/VI_Landsat/'
    output_folder = current_dir + '/2_Output/Landsat_recovery_resistance/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. Get File List
    filenames = [f for f in os.listdir(input_folder) if f.startswith('N') and f.endswith('.csv')]
    IDs = sorted([float(name.split('_')[-1].split('.csv')[0]) for name in filenames])

    # Parameters
    BANDS_YEAR = 12
    YR_NUM = 24

    # Detrending Parameter:
    # Window size for rolling mean. 24 time steps = 2 years (assuming monthly/12 bands).
    # This removes low-frequency trends to isolate short-term extremes.
    ROLLING_WINDOW = 12*7

    BOUNDARY_SD = 0.5  # Start/End definition
    CORE_SD_LEVELS = [1, 1.5, 2, 2.5]  # Disturbance intensity levels

    for id_val in IDs:
        print(f"Processing ID: {id_val}...")

        # --- Load and Preprocess Data (Done ONCE per file) ---
        try:
            df_i = pd.read_csv(f"{input_folder}NDVI_8d_{id_val}.csv").astype(float)
        except:
            continue

        vis = np.tanh(df_i.iloc[:, :-2].values)
        vis = fill_nan_with_climatology(vis, BANDS_YEAR)

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
                dVI_list =[]

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
            suffix = f"{id_val}_smith_{n_sd}sd.npy"
            np.save(f"{output_folder}resistance_{suffix}", resistance_out)
            np.save(f"{output_folder}resilience_{suffix}", resilience_out)
            np.save(f"{output_folder}dVI_{suffix}", dVI_out)
            np.save(f"{output_folder}extreme_events_{suffix}", extreme_events_out)

            print(
                f"  -> n={n_sd}SD: {np.sum(~np.isnan(resistance_out))} pixels affected. Avg events/pixel: {avg_events:.4f}")