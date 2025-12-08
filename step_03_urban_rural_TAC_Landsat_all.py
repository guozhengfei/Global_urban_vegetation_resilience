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

def ar1_series_5yr(array):
    import numpy.ma as ma
    return ma.corrcoef(ma.masked_invalid(array[:-1]), ma.masked_invalid(array[1:]))[0, 1]


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    folder = current_dir + '/2_Output/VI_Landsat/'
    urban_folder = current_dir + '/2_Output/nanFrac_urban_label_Landsat/'
    filenames = os.listdir(folder)

    names = []
    for name in filenames:
        if name.startswith('N'):
            names.append(name)

    IDs = []
    for name in names:
        id = float(name.split('_')[-1].split('.csv')[0])
        IDs.append(id)

    TAC = []
    IDs_num = []
    IDs = np.sort(IDs)
    for id in IDs:
        df_i = pd.read_csv(folder+'NDVI_8d_'+str(id)+'.csv').astype(float)
        vis = df_i.iloc[:,:-2].values
        vis[:, 13 * 12:13 * 12 + 12] = (vis[:, 12 * 12:12 * 12 + 12] + vis[:, 14 * 12:14 * 12 + 12]) / 2
        # plt.figure(); plt.hist(np.sum(np.isnan(vis),axis=1)/288,50)

        # fill with climatenoligy
        yr_num = 24
        bands_year = 12
        vis = fill_nan_with_climatology(vis)
        vis = fill_nan_with_climatology(vis)

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
        tac = np.array(results)
        tac[tac<0]=np.nan

        urban_nonfrac_labels = np.load(urban_folder + 'label_' + str(id) + '.npy')
        urban_lab = urban_nonfrac_labels[0, :]
        nan_frac = urban_nonfrac_labels[1, :]

        tac_tmp_inner = np.nanmean(tac[urban_lab == 2], axis=0)
        tac_tmp_sub = np.nanmean(tac[urban_lab == 1], axis=0)
        tac_tmp_rural = np.nanmean(tac[urban_lab == 0], axis=0)
        # plt.figure(); plt.plot(tac_tmp_inner); plt.plot(tac_tmp_sub); plt.plot(tac_tmp_rural)
        TAC.append([tac_tmp_inner, tac_tmp_sub, tac_tmp_rural])
        IDs_num.append(float(id))
        print(id)

    TAC_arr = np.array(TAC)
    IDs_num = np.array(IDs_num)
    output_file = current_dir + '/2_Output/tac_landsat_city_3zones_all.npz'
    np.savez(output_file, array1=TAC_arr, array2=IDs_num)


