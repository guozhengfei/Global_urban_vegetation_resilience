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
    for id in IDs:
        df_i = pd.read_csv(folder+'NDVI_8d_'+str(id)+'.csv').astype(float)
        vis = df_i.iloc[:,:-2].values
        vis[:, 13 * 12:13 * 12 + 12] = (vis[:, 12 * 12:12 * 12 + 12] + vis[:, 14 * 12:14 * 12 + 12]) / 2
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

        # remove long-term treng
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
        with mp.Pool(12) as pool:
            results = list(pool.map(ar1_series_5yr, divided_arrays))
        ar1_res_5sg = np.array(results)

        output_file = current_dir + '/2_Output/tac_Landsat/' + 'tac_' + str(id) + '.npy'
        np.save(output_file, ar1_res_5sg)
        print(id)

        TAC_mean = np.nanmean(ar1_res_5sg, axis=1)
        vi_mean = np.nanmean(vis, axis=1)

        plt.figure(); plt.plot(TAC_mean,vi_mean,'.')

