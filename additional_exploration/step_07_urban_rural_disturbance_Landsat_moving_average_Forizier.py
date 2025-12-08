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

    IDs = np.sort(IDs)[1:]
    for id in IDs:
        df_i = pd.read_csv(folder+'NDVI_8d_'+str(id)+'.csv').astype(float)
        vis = df_i.iloc[:,:-2].values
        ta_i = tf.imread(Data_folder + 'Ta_monthly_' + str(id) + '.tif')#[:, :, :264 * 2]
        ta_i = np.nanmedian(np.nanmedian(ta_i,axis=0),axis=0)
        ta_i_rsp = np.reshape(ta_i,(int(ta_i.shape[0]/12),12))
        ta_i_sea = np.nanmedian(ta_i_rsp,axis=0)-273.15

        gs_mask = ta_i_sea>6 # growing season mask

        # fill with climatenoligy
        yr_num = 24
        bands_year = 12

        vis=fill_with_climatology(vis, yr_num, bands_year)
        vis=fill_with_climatology(vis, yr_num, bands_year)
        # plt.figure(); plt.plot(vis[855,:])

        # growing season mean
        ser = vis*1
        gsm = [] # growing season mean
        for year in range(yr_num):
            st = year * bands_year
            ed = st + bands_year
            evi_gs = np.nanmean(ser[:, st:ed][:,gs_mask], axis=1)
            gsm.append(evi_gs)
        gsm_arr = np.array(gsm).T

        mean_all = np.nanmean(gsm_arr,axis=1)
        std = np.nanstd(gsm_arr,axis=1)

        # plt.figure(); plt.plot(gsm_arr[855,:])

        # Identify extreme events for each pixel
        threshold = mean_all - 1* std  # threshold for extreme events
        extreme_events = gsm_arr < threshold[:, np.newaxis]  # boolean mask for extreme events
        
        # Initialize arrays to store resistance and resilience indicators
        resistance_values = np.full(gsm_arr.shape[0], np.nan)
        resilience_values = np.full(gsm_arr.shape[0], np.nan)
        
        # Process each pixel
        for pixel_idx in range(gsm_arr.shape[0]):
            pixel_data = gsm_arr[pixel_idx, :]
            pixel_extreme = extreme_events[pixel_idx, :]
            pixel_threshold = threshold[pixel_idx]
            pixel_mean = mean_all[pixel_idx]
            
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
            
            for start, end in extreme_periods:
                # Resistance: minimum value during the extreme event relative to pre-event baseline
                if start > 0:
                    pre_event_baseline = np.nanmean(pixel_data[max(0, start-3):start])
                else:
                    pre_event_baseline = pixel_mean
                
                min_during_event = np.nanmin(pixel_data[start:end+1])
                resistance = min_during_event / pre_event_baseline if pre_event_baseline != 0 else np.nan
                pixel_resistance.append(resistance)
                
                # Resilience: recovery rate after the extreme event
                post_event_start = end + 1
                if post_event_start < len(pixel_data):
                    # Find recovery period (up to 5 years or until next extreme event)
                    recovery_end = min(post_event_start + 5, len(pixel_data))
                    
                    # Check if there's another extreme event in recovery period
                    next_extreme_start = None
                    for i in range(post_event_start, recovery_end):
                        if pixel_extreme[i]:
                            next_extreme_start = i
                            break
                    
                    if next_extreme_start is not None:
                        recovery_end = next_extreme_start
                    
                    if recovery_end > post_event_start:
                        recovery_data = pixel_data[post_event_start:recovery_end]
                        if len(recovery_data) > 0:
                            # Calculate resilience as the slope of recovery
                            years = np.arange(len(recovery_data))
                            if len(years) > 1:
                                # Linear regression to get recovery slope
                                A = np.vstack([years, np.ones(len(years))]).T
                                try:
                                    slope, intercept = np.linalg.lstsq(A, recovery_data, rcond=None)[0]
                                    resilience = slope
                                except:
                                    resilience = np.nan
                            else:
                                resilience = np.nan
                        else:
                            resilience = np.nan
                    else:
                        resilience = np.nan
                else:
                    resilience = np.nan
                
                pixel_resilience.append(resilience)
            
            # Store mean resistance and resilience for this pixel
            if len(pixel_resistance) > 0:
                resistance_values[pixel_idx] = np.nanmean(pixel_resistance)
            if len(pixel_resilience) > 0:
                resilience_values[pixel_idx] = np.nanmean(pixel_resilience)
        
        # Save results
        output_file_resistance = current_dir + '/2_Output/Landsat_recovery_resistance/' + 'resistance_' + str(id) + '_forizier.npy'
        output_file_resilience = current_dir + '/2_Output/Landsat_recovery_resistance/' + 'resilience_' + str(id) + '_forizier.npy'
        # output_file_extreme_events = current_dir + '/2_Output/tac_Landsat/' + 'extreme_events_' + str(id) + '.npy'
        
        np.save(output_file_resistance, resistance_values)
        np.save(output_file_resilience, resilience_values)
        # np.save(output_file_extreme_events, extreme_events)
        
        print(f"Processed ID {id}: {np.sum(np.any(extreme_events, axis=1))} pixels with extreme events")
