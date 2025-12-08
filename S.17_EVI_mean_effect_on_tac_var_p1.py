import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;
matplotlib.use('Qt5Agg')
import warnings
warnings.filterwarnings("ignore")
from scipy.ndimage import convolve1d
import os
import multiprocess as mp


def ar1_series_5yr(array):
    yrs = 5  # 3,5,7
    import pandas as pd
    import numpy.ma as ma
    def calc_ar1(x):
        return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0, 1]

    bands_year = 12
    t = bands_year * yrs

    ar1 = pd.Series(array).rolling(t, min_periods=6 * yrs, center=True).apply(calc_ar1).values
    return ar1

def var_series_5yr(array):
    yrs = 5  # 3,5,7
    import pandas as pd
    import numpy.ma as ma
    def calc_variance(x):
        return ma.var(ma.masked_invalid(x))  # 计算窗口内的方差，忽略无效值

    bands_year = 12
    t = bands_year * yrs

    variance = pd.Series(array).rolling(t,  center=True).apply(calc_variance).values
    return variance

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
    id = '6.0'
    df_i = pd.read_csv(folder+'NDVI_8d_'+str(id)+'.csv').astype(float)
    vis = df_i.iloc[:,:-2].values

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

    # Create vis2
    vis2 = 0.5 * vis

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10*0.6, 6*0.6),sharey=True,sharex=True)

    # Plot first subplot (original vis)
    mean_vis = np.nanmean(vis, axis=0)
    std_vis = np.nanstd(vis, axis=0)
    time = np.linspace(2000, 2024, 288)
    
    ax1.plot(time, mean_vis, 'k-', label='Mean')
    ax1.fill_between(time, 
                     mean_vis - std_vis,
                     mean_vis + std_vis,
                     alpha=0.3,
                     color='gray',
                     label='± SD')
    ax1.set_ylabel('VIs Value')
    ax1.legend()

    # Plot second subplot (vis2)
    mean_vis2 = np.nanmean(vis2, axis=0)
    std_vis2 = np.nanstd(vis2, axis=0)
    
    ax2.plot(time, mean_vis2, 'k-', label='Mean')
    ax2.fill_between(time, 
                     mean_vis2 - std_vis2,
                     mean_vis2 + std_vis2,
                     alpha=0.3,
                     color='gray',
                     label='± SD')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('VIs Value (×0.5)')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(current_dir + '/4_Figures/FigS04_VI_and_its_half.png', dpi=900)

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
    # plt.figure(); plt.plot(np.nanmean(rm_offline, axis=0))

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
    res[np.isnan(res)] = 0
    # plt.figure(); plt.plot(np.nanmean(res,axis=0))

    divided_arrays = [row for row in res]
    with mp.Pool(8) as pool:
        results = list(pool.map(ar1_series_5yr, divided_arrays))
    ar1_res_5sg = np.array(results)

    with mp.Pool(8) as pool:
        results = list(pool.map(var_series_5yr, divided_arrays))
    var_res_5sg = np.array(results)

    # output_file = current_dir + '/2_Output/tac_Landsat/' + 'tac_' + str(id) + '.npy'
    # np.save(output_file, ar1_res_5sg)
    # print(id)


    #######################################################################

    # remove long-term mean
    ser = vis2 * 1
    EVI_yr = np.zeros_like(ser)
    for year in range(yr_num):
        st = year * bands_year
        ed = st + bands_year
        evi_year = np.nanmean(ser[:, st:ed], axis=1)
        evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
        EVI_yr[:, st:ed] = evi_year_rep
    rm_offline = ser - EVI_yr
    # plt.figure(); plt.plot(np.nanmean(rm_offline, axis=0))

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
    res[np.isnan(res)] = 0
    # plt.figure(); plt.plot(np.nanmean(res,axis=0))

    divided_arrays = [row for row in res]
    with mp.Pool(8) as pool:
        results = list(pool.map(ar1_series_5yr, divided_arrays))
    ar1_res_5sg2 = np.array(results)

    with mp.Pool(8) as pool:
        results = list(pool.map(var_series_5yr, divided_arrays))
    var_res_5sg2 = np.array(results)

    ar1_res_5sg2[ar1_res_5sg2<0.02]=np.nan
    ar1_res_5sg[ar1_res_5sg<0.02]=np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 0.6, 4.5 * 0.6))
    ax1.plot(ar1_res_5sg.flatten(),ar1_res_5sg2.flatten(),'o',mfc='None')
    ax1.set_xlim([0,0.7]); ax1.set_ylim([0,0.7])
    ax1.plot([0, 0.7],[0, 0.7],'r-')
    ax2.plot(var_res_5sg.flatten(),var_res_5sg2.flatten(),'o',mfc='None')
    ax2.set_xlim([0, 0.03]); ax2.set_ylim([0, 0.03])
    ax2.plot([0, 0.03], [0, 0.03], 'r-')

    fig.tight_layout()
    plt.savefig(current_dir + '/4_Figures/FigS04_VI_and_its_half_scatter.png', dpi=900)
