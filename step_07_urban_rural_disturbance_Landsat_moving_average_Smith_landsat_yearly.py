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
    ROLLING_WINDOW = 7

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
        gs_mask = vi_seasonal > vi_threshold
        vis_monthly[:,:,~gs_mask]=np.nan
        vis_yearly = np.nanmean(vis_monthly,axis=2)


        # --- De-trend (Rolling Mean) ---
        # Convert to DataFrame to leverage optimized rolling function
        # Transpose to (Time, Pixels) because rolling works on index
        df_deseas = pd.DataFrame(vis_yearly.T)

        # Calculate Rolling Mean (Trend)
        # center=True ensures the trend is aligned with the event
        # min_periods=1 ensures edges are handled
        rolling_trend = df_deseas.rolling(window=ROLLING_WINDOW, center=True, min_periods=1).mean()

        # Calculate Residuals (Anomaly - Trend)
        residuals = vis_yearly - rolling_trend.values.T

        # Add Static Mean back to standardize the baseline for thresholding
        vis_mean_pixel = np.nanmean(vis, axis=1, keepdims=True)
        vis_mean_pixel = np.tile(vis_mean_pixel,YR_NUM)
        gsm_arr = residuals

        # --- Statistics ---
        mean_all = np.nanmean(gsm_arr, axis=1)
        std_all = np.nanstd(gsm_arr, axis=1)
        n_pixels, n_times = gsm_arr.shape

        # --- Loop through n = 1, 2, 3 ---
        for n_sd in CORE_SD_LEVELS:
            # Define Thresholds
            core_thresholds = mean_all - (n_sd * std_all)
            core_thresholds_2d = np.tile(core_thresholds, (YR_NUM, 1)).T


            # Output Arrays
            # Output Arrays
            dVI_out = gsm_arr + np.nan
            extreme_events_out = np.zeros((n_pixels, n_times), dtype=bool)

            disturbance_label = residuals < core_thresholds_2d
            dVI_out[disturbance_label] = residuals[disturbance_label] / vis_mean_pixel[disturbance_label]
            dVI_out = np.nanmin(dVI_out, axis=1)

            # Calculate average events
            total_events = np.sum(disturbance_label)
            avg_events = total_events / n_pixels
            extreme_events_out[disturbance_label] = True
            extreme_events_out_month = np.repeat(extreme_events_out, 12, axis=1)

            # Save
            suffix = f"{id_val}_smith_{n_sd}sd.npy"

            np.save(f"{output_folder}dVI_{suffix}", dVI_out)
            np.save(f"{output_folder}extreme_events_{suffix}", extreme_events_out)

            print(
                f" Landsat {id} -> n={n_sd}SD: {np.sum(~np.isnan(dVI_out))} pixels affected. Avg events/pixel: {avg_events:.4f}")