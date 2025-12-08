import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
# import matplotlib;
# matplotlib.use('Qt5Agg')
import warnings
warnings.filterwarnings("ignore")
from scipy.ndimage import convolve1d
import os
import multiprocess as mp


def cal_slope(y):
    # Create x array (years)
    x = np.arange(len(y))
    
    # Remove NaN values
    mask = ~np.isnan(y)
    if np.sum(mask) < 3:  # Need at least 3 points for meaningful trend
        return np.nan
    
    x_valid = x[mask]
    y_valid = y[mask]
    
    try:
        # Calculate linear regression
        A = np.vstack([x_valid, np.ones(len(x_valid))]).T
        slope, _ = np.linalg.lstsq(A, y_valid, rcond=None)[0]
        return slope
    except:
        return np.nan


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

    IDs = np.sort(IDs)
    slopes = []
    for id in IDs:
        df_i = pd.read_csv(folder+'NDVI_8d_'+str(id)+'.csv').astype(float)
        vis = df_i.iloc[:,:-2].values
        # plt.figure(); plt.hist(np.sum(np.isnan(vis),axis=1)/288,50)

        # fill with climatenoligy
        yr_num = 24
        bands_year = 12
        Evi_sea_rep = np.zeros_like(vis)
        for yr in range(yr_num):
            start_index = (yr - 2) * bands_year
            if start_index < 0: start_index = 0
            end_index = (yr + 3) * bands_year
            if end_index > bands_year * yr_num: end_index = bands_year * yr_num
            data_i = vis[:, start_index:end_index]
            Evi_sea = np.nanmean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)),
                              axis=1)
            Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea
        vis[np.isnan(vis)] = Evi_sea_rep[np.isnan(vis)]
        vis_yearly = np.reshape(vis, (vis.shape[0], int(vis.shape[1] / bands_year), bands_year))
        vis_yearly = np.nanmean(vis_yearly, axis=2)
        vis_mean = np.nanmean(vis_yearly,axis=1)


        divided_arrays = [row for row in vis_yearly]
        with mp.Pool(6) as pool:
            results = list(pool.map(cal_slope, divided_arrays))
        trend_i = np.array([results,vis_mean])
        output_file = current_dir + '/2_Output/VI_trend/' + 'ndvi_trend_and_mean' + str(id) + '.npy'
        np.save(output_file, trend_i)
        print(id)
