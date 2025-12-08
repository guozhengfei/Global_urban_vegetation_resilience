import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib;
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.signal as ss
import warnings
warnings.filterwarnings("ignore")
from scipy.ndimage import convolve1d
import os
import multiprocess as mp

def fill_with_climatology(vis,yr_num,bands_year):
    Evi_sea_rep = np.zeros_like(vis)
    for yr in range(yr_num):
        start_index = (yr - 3) * bands_year
        if start_index < 0: start_index = 0
        end_index = (yr + 4) * bands_year
        if end_index > bands_year * yr_num: end_index = bands_year * yr_num
        data_i = vis[:, start_index:end_index]
        Evi_sea = np.nanmean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)),
                             axis=1)
        Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea
    vis[np.isnan(vis)] = Evi_sea_rep[np.isnan(vis)]
    return vis

def calc_window_diff(array):
    window_size = 18  # 10-month window
    
    # Initialize output array with NaN
    window_means = np.full_like(array, np.nan, dtype=float)
    
    # Calculate moving average
    for i in range(window_size-1, len(array)):
        window = array[i-window_size+1:i+1]
        window_means[i] = np.nanmean(window[9:])-np.nanmean(window[0:9])
    return window_means

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    folder = current_dir + '/2_Output/VI_Landsat/'
    # urban_folder = current_dir+'/1_Input/urban_area/'
    filenames = os.listdir(folder)

    names = []
    for name in filenames:
        if name.startswith('N'):
            names.append(name)

    IDs = []
    for name in names:
        id = float(name.split('_')[-1].split('.csv')[0])
        IDs.append(id)

    Data_folder = os.path.dirname(current_dir) + '/urban_env_data/Ta_ear5Land_urban/'

    IDs = np.sort(IDs)
    even_nums = []
    for id in IDs[1:]:
        df_i = pd.read_csv(folder+'NDVI_8d_'+str(id)+'.csv').astype(float)
        vis = df_i.iloc[:,:-2].values
        ta_i = tf.imread(Data_folder + 'Ta_monthly_' + str(id) + '.tif')#[:, :, :264 * 2]
        ta_i = np.nanmedian(np.nanmedian(ta_i,axis=0),axis=0)
        ta_i_rsp = np.reshape(ta_i,(int(ta_i.shape[0]/12),12))
        ta_i_sea = np.nanmedian(ta_i_rsp,axis=0)-273.15
        gs_mask = ta_i_sea>6
        gs_mask = np.repeat(gs_mask[np.newaxis],24,axis=0).reshape(-1)

        # fill with climatenoligy
        yr_num = 24
        bands_year = 12

        vis=fill_with_climatology(vis, yr_num, bands_year)
        vis=fill_with_climatology(vis, yr_num, bands_year)

        ser = vis*1
        # remove seasonality
        Evi_sea_rep = np.zeros_like(ser)
        for yr in range(yr_num):
            start_index = (yr - 3) * bands_year
            if start_index < 0: start_index = 0
            end_index = (yr + 4) * bands_year
            if end_index > bands_year * yr_num: end_index = bands_year * yr_num
            data_i = ser[:, start_index:end_index]
            Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)), axis=1)
            Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea

        res0 = ser - Evi_sea_rep
        test= np.nanstd(res0,axis=1, keepdims=True)
        res = res0#/np.nanmean(vis,axis=1, keepdims=True)
        res[:,~gs_mask] = 0
        res[np.isnan(res)] = 0
        gsm_arr = ss.savgol_filter(res, 18, 1, mode='nearest', axis=1)
        # gsm_arr = ss.savgol_filter(gsm_arr, 7, 1, mode='nearest', axis=1)
        gsm_arr = gsm_arr + np.nanmean(vis,axis=1, keepdims=True)

        mean_all = np.nanmean(gsm_arr, axis=1)
        std = np.nanstd(gsm_arr, axis=1)

        # plt.figure(); plt.plot(res[855,:])
        # plt.figure(); plt.plot(vis[855,:])

        # Identify extreme events for each pixel
        threshold = mean_all - 3.0 * std  # threshold for extreme events
        extreme_events = gsm_arr < threshold[:, np.newaxis]  # boolean mask for extreme events

        # Initialize arrays to store resistance and resilience indicators
        resistance_values = np.full(gsm_arr.shape[0], np.nan)
        resilience_values = np.full(gsm_arr.shape[0], np.nan)
        dt_VI_values = np.full(gsm_arr.shape[0], np.nan)


        # Process each pixel
        for pixel_idx in range(gsm_arr.shape[0]):
            pixel_data = gsm_arr[pixel_idx, :]
            pixel_extreme = extreme_events[pixel_idx, :]
            pixel_threshold = threshold[pixel_idx]
            pixel_mean = mean_all[pixel_idx]
            # plt.figure(); plt.plot(pixel_data)

            # Find continuous extreme event periods
            extreme_periods = []
            start_idx = None

            for year_idx in range(len(pixel_extreme)):
                if pixel_extreme[year_idx] and start_idx is None:
                    start_idx = year_idx
                elif not pixel_extreme[year_idx] and start_idx is not None:
                    extreme_periods.append((start_idx, year_idx - 1))
                    start_idx = None

            # Handle case where extreme event continues to the end
            if start_idx is not None:
                extreme_periods.append((start_idx, len(pixel_extreme) - 1))

            # Calculate resistance and resilience for each extreme event period
            pixel_resistance = []
            pixel_resilience = []
            pixel_dt_VI = []

            for start, end in extreme_periods:
                # Resistance: minimum value during the extreme event relative to pre-event baseline
                if end - start<4: continue
                if start > 0:
                    pre_event_baseline = np.nanmax(pixel_data[max(0, start - 3*12):start])
                else:
                    pre_event_baseline = pixel_mean

                resist_start = start
                min_during_event = np.nanmin(pixel_data[start:end])
                resist_end = start + np.where(pixel_data[start:end]==min_during_event)[0]
                try:
                    resist_data = pixel_data[start:resist_end[0]]

                    months0 = np.arange(len(resist_data))
                    # Linear regression to get recovery slope
                    A0 = np.vstack([months0, np.ones(len(months0))]).T

                    slope0, intercept0 = np.linalg.lstsq(A0, resist_data, rcond=None)[0]
                    resistance = slope0
                    dt_VI = resist_data.min() - resist_data.max()

                except:
                    resistance = np.nan
                    dt_VI = np.nan

                # resistance = min_during_event / pre_event_baseline if pre_event_baseline != 0 else np.nan
                # resistance = pre_event_baseline- min_during_event  if pre_event_baseline != 0 else np.nan
                pixel_resistance.append(resistance)
                pixel_dt_VI.append(dt_VI)


                # Resilience: recovery rate after the extreme event
                post_event_start = start + np.where(pixel_data[start:end]==min_during_event)[0]

                recovery_end =  end
                # recovery_end =  post_event_start + 5


                try:
                    recovery_data = pixel_data[post_event_start[0]:recovery_end]

                    months = np.arange(len(recovery_data))
                    # Linear regression to get recovery slope
                    A = np.vstack([months, np.ones(len(months))]).T

                    slope, intercept = np.linalg.lstsq(A, recovery_data, rcond=None)[0]
                    resilience = slope
                except:
                    resilience = np.nan


                pixel_resilience.append(resilience)

            # Store mean resistance and resilience for this pixel
            if len(pixel_resistance) > 0:
                resistance_values[pixel_idx] = np.nanmean(pixel_resistance)
            if len(pixel_resilience) > 0:
                resilience_values[pixel_idx] = np.nanmean(pixel_resilience)
            if len(pixel_dt_VI) > 0:
                dt_VI_values[pixel_idx] = np.nanmean(pixel_dt_VI)


        # Save results
        output_file_resistance = current_dir + '/2_Output/Landsat_recovery_resistance/' + 'resistance_' + str(
            id) + '_smith_10sd.npy'
        output_file_resilience = current_dir + '/2_Output/Landsat_recovery_resistance/' + 'resilience_' + str(
            id) + '_smith_10sd.npy'
        output_file_extreme_events = current_dir + '/2_Output/tac_Landsat/' + 'extreme_events_' + str(id) + '.npy'


        np.save(output_file_resistance, resistance_values)
        np.save(output_file_resilience, resilience_values)
        np.save(output_file_extreme_events, extreme_events)

        even_nums.append(len(pixel_resistance))
        np.array(even_nums).mean()

        print(f"Processed ID {id}: {np.sum(np.any(extreme_events, axis=1))} pixels with extreme events")