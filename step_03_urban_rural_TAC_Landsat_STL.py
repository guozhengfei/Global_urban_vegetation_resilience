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

def fill_with_climatology(vis,yr_num,bands_year):
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
    return vis
def ar1_series_5yr(array):
    yrs = 5  # 3,5,7
    import pandas as pd
    import numpy.ma as ma
    def calc_ar1(x):
        return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0, 1]
    from statsmodels.tsa.seasonal import STL
    stl = STL(array.T, period=12,seasonal=13,robust=True)  # 13 for monthly data
    result = stl.fit()
    res = result.resid

    bands_year = 12

    t = bands_year * yrs
    ar1 = pd.Series(res).rolling(t, min_periods=6 * yrs, center=True).apply(calc_ar1).values
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
        # plt.figure(); plt.hist(np.sum(np.isnan(vis),axis=1)/288,50)
        yr_num = 24
        bands_year = 12

        #fill nan with climatology; twice
        vis = fill_with_climatology(vis,yr_num,bands_year)
        vis = fill_with_climatology(vis, yr_num, bands_year)

        divided_arrays = [row for row in vis]
        with mp.Pool(6) as pool:
            results = list(pool.map(ar1_series_5yr, divided_arrays))
        ar1_res_5sg = np.array(results)
        output_file = current_dir + '/2_Output/tac_Landsat/' + 'tac_STL' + str(id) + '.npy'
        np.save(output_file, ar1_res_5sg)
        print(id)


