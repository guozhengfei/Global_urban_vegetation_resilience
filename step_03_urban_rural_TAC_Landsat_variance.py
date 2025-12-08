import numpy as np
import pandas as pd
import os
import multiprocess as mp
import matplotlib.pyplot as plt
import matplotlib;
matplotlib.use('Qt5Agg')
def var_series_5yr(array):
    yrs = 5
    bands_year = 12
    t = bands_year * yrs

    def calc_var(x):
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return np.nan

        var = np.var(x)
        return var

    # 使用NumPy的滑动窗口
    result = np.full(array.shape, np.nan)
    for i in range(len(array)):
        left = max(0, i - t // 2)
        right = min(len(array), i + t // 2 + t % 2)
        window = array[left:right]
        result[i] = calc_var(window)
    return result

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    folder = current_dir + '/2_Output/VI_Landsat/'
    filenames = [name for name in os.listdir(folder) if name.startswith('N')]
    IDs = np.sort([float(name.split('_')[-1].split('.csv')[0]) for name in filenames])

    yr_num = 24
    bands_year = 12

    # 动态分配进程数
    cpu_count = mp.cpu_count()
    pool_size = min(cpu_count, 12)  # 可根据实际情况调整

    for id in IDs:
        df_i = pd.read_csv(f"{folder}NDVI_8d_{id}.csv").astype(float)
        vis = df_i.iloc[:, :-2].values

        # 用气候学均值填补缺失
        Evi_sea_rep = np.zeros_like(vis)
        for yr in range(yr_num):
            start_index = max(0, (yr - 2) * bands_year)
            end_index = min(bands_year * yr_num, (yr + 3) * bands_year)
            data_i = vis[:, start_index:end_index]
            Evi_sea = np.nanmean(data_i.reshape(data_i.shape[0], -1, bands_year), axis=1)
            Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea
        vis[np.isnan(vis)] = Evi_sea_rep[np.isnan(vis)]

        # 去除长期均值
        ser = vis.copy()
        EVI_yr = np.zeros_like(ser)
        for year in range(yr_num):
            st, ed = year * bands_year, (year + 1) * bands_year
            evi_year = np.nanmean(ser[:, st:ed], axis=1)
            EVI_yr[:, st:ed] = evi_year[:, None]
        rm_offline = ser - EVI_yr

        # 去除季节性
        Evi_sea_rep = np.zeros_like(rm_offline)
        for yr in range(yr_num):
            start_index = max(0, (yr - 3) * bands_year)
            end_index = min(bands_year * yr_num, (yr + 4) * bands_year)
            data_i = rm_offline[:, start_index:end_index]
            Evi_sea = np.mean(data_i.reshape(data_i.shape[0], -1, bands_year), axis=1)
            Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea
        res = rm_offline - Evi_sea_rep
        res[np.isnan(res)] = 0
        res2 = res+np.nanmean(vis,axis=1,keepdims=True)

        # 多进程计算CV
        with mp.Pool(pool_size) as pool:
            results = pool.map(var_series_5yr, res2)
        ar1_res_5sg = np.array(results)

        output_file = f"{current_dir}/2_Output/tac_Landsat/tac_{id}_cv.npy"
        np.save(output_file, ar1_res_5sg)
        print(id)

        TAC_mean = np.nanmean(ar1_res_5sg, axis=1)
        vi_mean = np.nanmean(vis, axis=1)

        plt.figure();
        plt.plot(TAC_mean, vi_mean, '.')
