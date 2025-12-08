import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;

#matplotlib.use('Qt5Agg')
import scipy.signal as ss
import multiprocess as mp
import os
import warnings
import cv2
warnings.filterwarnings("ignore")
import numpy as np
import scipy.stats as st
import pandas as pd

def extend_edge(array):
    array1 = array * 1
    array1[1:, :] = array[:-1, :]  # Shift elements up
    array2 = array * 1
    array2[:-1, :] = array[1:, :]  # Shift elements down
    array3 = array * 1
    array3[:, 1:] = array[:, :-1]  # Shift elements left
    array4 = array * 1
    array4[:, :-1] = array[:, 1:]  # Shift elements right
    array5 = array * 1
    array5[:-1, 1:] = array[1:, :-1]  # Shift elements up-left
    array6 = array * 1
    array6[:-1, :-1] = array[1:, 1:]  # Shift elements up-right
    array7 = array * 1
    array7[1:, 1:] = array[:-1, :-1]  # Shift elements down-right
    array8 = array * 1
    array8[1:, :-1] = array[:-1, 1:]  # Shift elements down-left
    stacked_array = np.stack([array1, array2, array3, array4, array5, array6, array7, array8], axis=0)
    result = np.nanmean(stacked_array, axis=0)
    result[result > 0] = 1
    return result


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    tac_folder = current_dir + '/2_Output/tac_500m/'
    urban_folder_2018 = current_dir + '/1_Input/urban/urban_2018_1000m/'
    urban_folder_1990 = current_dir + '/1_Input/urban/urban_1990_250m/'
    # df_pdsi = pd.read_csv(current_dir + '/2_Output/pdsi_2001_2022.csv')
    # df_tmmx = pd.read_csv(current_dir + '/2_Output/tmmx_2001_2022.csv')

    filenames = os.listdir(tac_folder)
    names = []
    for name in filenames:
        if name.endswith('_v2.npy') and name.startswith('t'):
            names.append(name)
    IDs = []
    for name in names:
        id = name.split('_')[1]#.split('.npy')[0]
        IDs.append(id)

    TAC = []
    IDs_num = []
    for id in IDs:
        tac = np.load(tac_folder + 'tac_' + id + '_v2.npy')[:,:,2:-2]
        # tac[tac<=0]=np.nan
        # plt.figure(); plt.hist(tac.reshape(-1),100)
        tac_spa_mean = np.nanmean(tac, axis=2)
        # plt.figure(); plt.imshow(tac_spa_mean)
        urban_2018 = tf.imread(urban_folder_2018 + 'fvc_' + id + '.tif').astype(float)
        urban_2018_rsz = cv2.resize(urban_2018, (tac.shape[1], tac.shape[0]),cv2.INTER_NEAREST)
        urban_2018_rsz[urban_2018_rsz != float(id)] = np.nan
        urban_1990 = tf.imread(urban_folder_1990 + 'urban_1990_' + id + '.tif').astype(float)
        urban_1990= cv2.resize(urban_1990, (tac.shape[1], tac.shape[0]),cv2.INTER_NEAREST)
        urban_1990[urban_1990 == 0] = np.nan

        rural_near = urban_2018_rsz * 1  # rural-urban interface
        for i in range(3*2):
            rural_near = extend_edge(rural_near)

        rural_bgr = rural_near * 1  # rural background
        for i in range(10*2):
            rural_bgr = extend_edge(rural_bgr)

        # URBAN 1990
        urban_1990_3d = np.tile(urban_1990[:, :, np.newaxis], tac.shape[2])

        tac_urban_1990_all = tac + urban_1990_3d - urban_1990_3d
        if np.sum(~np.isnan(tac_urban_1990_all))<100: continue
        tac_urban_1990 = np.nanmean(np.nanmean(tac_urban_1990_all, axis=0), axis=0)

        # plt.figure();
        # plt.plot(np.linspace(2004, 2021 , 18), tac_urban_1990)

        # URBAN 1990-2018
        urban_1990_v2 = urban_1990 * 1
        urban_1990_v2[np.isnan(urban_1990_v2)] = 0
        urban_1990_v2[urban_1990_v2 != 0] = np.nan
        urban_1990_2018 = rural_near + urban_1990_v2
        # plt.figure(); plt.imshow(urban_1990_2018)
        urban_1990_2018_3d = np.tile(urban_1990_2018[:, :, np.newaxis], tac.shape[2])
        tac_urban_2018_all = tac + urban_1990_2018_3d - urban_1990_2018_3d
        if np.sum(~np.isnan(tac_urban_2018_all)) < 100: continue
        tac_urban_1990_2018 = np.nanmean(np.nanmean(tac + urban_1990_2018_3d - urban_1990_2018_3d, axis=0), axis=0)
        # plt.plot(np.linspace(2004, 2021 , 18), tac_urban_1990_2018)

        # RURAL-BACKGROUND
        rural_near_v2 = rural_near * 1
        rural_near_v2[np.isnan(rural_near_v2)] = 0
        rural_near_v2[rural_near_v2 != 0] = np.nan
        urban_rural_bgr = rural_bgr + rural_near_v2
        # plt.figure(); plt.imshow(urban_rural_bgr,alpha=0.3)
        urban_rural_bgr_3d = np.tile(urban_rural_bgr[:, :, np.newaxis], tac.shape[2])
        tac_rural_all = tac + urban_rural_bgr_3d - urban_rural_bgr_3d
        if np.sum(~np.isnan(tac_rural_all)) < 100: continue
        tac_rural_bgr = np.nanmean(np.nanmean(tac + urban_rural_bgr_3d - urban_rural_bgr_3d, axis=0), axis=0)
        # plt.plot(np.linspace(2004, 2021, 18),tac_rural_bgr)

        TAC.append([tac_urban_1990, tac_urban_1990_2018, tac_rural_bgr])
        IDs_num.append(float(id))
        print(id)
    TAC_arr = np.array(TAC)

    IDs_num = np.array(IDs_num)
    output_file = current_dir + '/2_Output/' + 'tac_nadir_city_3zones_modis_month' + '.npz'
    np.savez(output_file, array1=TAC_arr, array2=IDs_num)
