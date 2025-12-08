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
    for id in IDs:
        df_i = pd.read_csv(folder+'NDVI_8d_'+str(id)+'.csv').astype(float)
        vis = df_i.iloc[:,:-2].values
        ta_i = tf.imread(Data_folder + 'Ta_monthly_' + str(id) + '.tif')#[:, :, :264 * 2]
        ta_i = np.nanmedian(np.nanmedian(ta_i,axis=0),axis=0)
        ta_i_rsp = np.reshape(ta_i,(int(ta_i.shape[0]/12),12))
        ta_i_sea = np.nanmedian(ta_i_rsp,axis=0)-273.15

        gs = ta_i_sea>6

        # fill with climatenoligy
        yr_num = 24
        bands_year = 12

        vis=fill_with_climatology(vis, yr_num, bands_year)
        vis=fill_with_climatology(vis, yr_num, bands_year)

        vis_mean = np.nanmean(vis,axis=0)
        vis_sea = np.nanmean(np.reshape(vis_mean,(int(vis_mean.shape[0]/12),12)),axis=0)

        plt.figure(); plt.plot(vis_sea)

        # remove long-term mean
        ser = vis*1
        EVI_yr = np.zeros_like(ser)
        for year in range(yr_num):
            st = year * bands_year
            ed = st + bands_year
            evi_year = np.nanmean(ser[:, st:ed], axis=1)
            evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
            EVI_yr[:, st:ed] = evi_year_rep
        rm_offline = ser - EVI_yr
        # plt.figure(); plt.plot(vis[15,:])
        plt.figure(); plt.plot(EVI_yr[15, :])

        # remove seasonality
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

        res = rm_offline - Evi_sea_rep
        res2 = ss.savgol_filter(res, 7, 1, mode='nearest', axis=1)

        res[np.isnan(res)] = 0
        # plt.figure(); plt.plot(res[855,:])

        divided_arrays = [row for row in res]
        with mp.Pool(6) as pool:
            results = list(pool.map(calc_window_diff, divided_arrays))
        window_diff = np.array(results)

        window_diff2 = ss.savgol_filter(window_diff, 7, 1, mode='nearest', axis=1)
        max_row = np.nanmax(window_diff2,axis=1)
        mask = max_row<0.01
        window_diff2[mask,:]=np.nan
        # example = window_diff2[~mask,:]

        output_file = current_dir + '/2_Output/tac_Landsat/' + 'res_move_ave__' + str(id) + '.npy'
        np.save(output_file, window_diff2)
        print(id)
