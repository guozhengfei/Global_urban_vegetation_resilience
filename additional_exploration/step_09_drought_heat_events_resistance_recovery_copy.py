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

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    folder = current_dir + '/2_Output/VI_Landsat/'
    drought_events_df= pd.read_csv(current_dir + '/2_Output/drought_events_results.csv')
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

    Data_folder = os.path.dirname(current_dir) + '/urban_env_data/Ta_ear5Land_urban/'

    IDs = np.sort(IDs)
    even_nums = []
    Resist_final = []
    for id in IDs[1:]:
        df_i = pd.read_csv(folder+'NDVI_8d_'+str(id)+'.csv').astype(float)
        vis = df_i.iloc[:,:-2].values
        ta_i = tf.imread(Data_folder + 'Ta_monthly_' + str(id) + '.tif')#[:, :, :264 * 2]
        ta_i = np.nanmedian(np.nanmedian(ta_i,axis=0),axis=0)
        ta_i_rsp = np.reshape(ta_i,(int(ta_i.shape[0]/12),12))
        ta_i_sea = np.nanmedian(ta_i_rsp,axis=0)-273.15
        gs_mask = ta_i_sea > 6
        gs_mask = np.repeat(gs_mask[np.newaxis], 24, axis=0).reshape(-1)
        urban_nonfrac_labels = np.load(urban_folder + 'label_' + str(id) + '.npy')
        urban_lab = urban_nonfrac_labels[0, :]

        # fill with climatenoligy
        yr_num = 24
        bands_year = 12

        vis=fill_with_climatology(vis, yr_num, bands_year)
        vis=fill_with_climatology(vis, yr_num, bands_year)
        vis_mean = np.nanmean(vis,axis=1)[np.newaxis]

        ser = vis*1
        # remove seasonality
        Evi_sea_rep = np.zeros_like(ser)
        for yr in range(yr_num):
            start_index = (yr - 4) * bands_year
            if start_index < 0: start_index = 0
            end_index = (yr + 5) * bands_year
            if end_index > bands_year * yr_num: end_index = bands_year * yr_num
            data_i = ser[:, start_index:end_index]
            Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)), axis=1)
            Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea

        res0 = ser - Evi_sea_rep
        res0[:,~gs_mask] = np.nan
        mask = (np.nanmean(vis, axis=1)<0.1) #
        droughts = drought_events_df.loc[drought_events_df['ID']==id,'drought_events'].values[0][1:-2].split('),')
        if len(droughts) < 2:  continue

        Resist =[]
        for event in droughts:
            start_index = int(event.split(',')[0].split('(')[1])
            end_index = int(event.split(',')[1][1:])
            min_during_event = np.nanmin(res0[:,start_index:end_index],axis=1)

            resist = min_during_event#/(np.nanstd(res0,axis=1)*3)
            Resist.append(resist)

        Resist_arr = np.array(Resist)
        Resist_arr[:,mask] = np.nan
        Resist_arr[Resist_arr>-0.02]=np.nan
        resist_final = np.nanmean(Resist_arr,axis=0)


        resis_inner = np.nanmean(resist_final[urban_lab == 2], axis=0)
        resis_sub = np.nanmean(resist_final[urban_lab == 1], axis=0)
        resis_rural = np.nanmean(resist_final[urban_lab == 0], axis=0)

        Resist_final.append([resis_inner,resis_sub,resis_rural])
        print(id)

    Resist_final_arr = np.array(Resist_final)

    np.nanmean(Resist_final_arr,axis=0)

    # vi_inner = np.nanmean(vis[urban_lab == 2,:], axis=0)
    # vi_sub = np.nanmean(vis[urban_lab == 1,:], axis=0)
    # vi = np.nanmean(vis[urban_lab == 0,:], axis=0)

    vi_inner = vis[urban_lab == 2,:]
    plt.figure(); plt.plot(np.nanmean(vi_inner,axis=0))

    vi_edge = vis[urban_lab == 1, :]
    plt.plot(np.nanmean(vi_edge,axis=0))

    vi_rural = vis[urban_lab == 0, :]
    plt.plot(np.nanmean(vi_rural, axis=0))
    #

