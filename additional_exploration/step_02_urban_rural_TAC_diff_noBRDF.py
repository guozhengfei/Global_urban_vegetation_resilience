import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;

matplotlib.use('Qt5Agg')
import scipy.signal as ss
import multiprocess as mp
import os
import warnings
import cv2
warnings.filterwarnings("ignore")
import numpy as np

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
    tac_folder = current_dir + '/2_Output/tac_500_noBRDF/'
    urban_folder_2018 = current_dir + '/1_Input/urban/urban_2018_1000m/'
    urban_folder_1990 = current_dir + '/1_Input/urban/urban_1990_250m/'
    urban_folder_2000 = current_dir + '/1_Input/urban/urban_2000_250m/'
    lcc_folder = current_dir + '/1_Input/land_cover_change_250m/'

    filenames = os.listdir(tac_folder)

    IDs = []
    for name in filenames:
        id = name.split('_')[-1].split('.npy')[0]
        IDs.append(id)

    TAC = []
    for id in IDs:
        tac = np.load(tac_folder + 'tac_' + id + '.npy')
        lcc = tf.imread(lcc_folder + 'lcc_' + id + '.tif')
        lcc[lcc > 0.05] = np.nan
        lcc = cv2.resize(lcc, (tac.shape[1], tac.shape[0]), cv2.INTER_LINEAR)
        lcc_rpt = np.tile(lcc[:,:,np.newaxis],(1,1,np.shape(tac)[2]))
        tac[np.isnan(lcc_rpt)]=np.nan

        tac_spa_mean = np.nanmean(tac, axis=2)
        urban_2018 = tf.imread(urban_folder_2018 + 'fvc_' + id + '.tif').astype(float)
        urban_2018_rsz = cv2.resize(urban_2018, (tac.shape[1], tac.shape[0]),cv2.INTER_NEAREST)
        urban_2018_rsz[urban_2018_rsz != float(id)] = np.nan
        urban_1990 = tf.imread(urban_folder_1990 + 'urban_1990_' + id + '.tif').astype(float)
        urban_1990= cv2.resize(urban_1990, (tac.shape[1], tac.shape[0]),cv2.INTER_NEAREST)
        urban_1990[urban_1990 == 0] = np.nan

        urban_2000 = tf.imread(urban_folder_2000 + 'urban_2000_' + id + '.tif').astype(float)
        urban_2000 = cv2.resize(urban_2000, (tac.shape[1], tac.shape[0]), cv2.INTER_NEAREST)
        urban_2000[urban_2000 == 0] = np.nan

        rural_near = urban_2018_rsz * 1  # rural-urban interface
        for i in range(7):
            rural_near = extend_edge(rural_near)

        rural_bgr = rural_near * 1  # rural background
        for i in range(10):
            rural_bgr = extend_edge(rural_bgr)

        # plt.figure();
        # plt.imshow(tac_spa_mean,cmap='RdYlBu_r',vmin=0.25,vmax=0.55)
        # plt.imshow(urban_1990, alpha=0.3)
        # plt.imshow(urban_2000, alpha=0.3)
        # plt.imshow(urban_2018_rsz, alpha=0.3)
        # plt.imshow(rural_near, alpha=0.3)
        # plt.imshow(rural_bgr, alpha=0.3)

        # URBAN 1990
        urban_1990_3d = np.tile(urban_1990[:, :, np.newaxis], tac.shape[2])
        tac_urban_1990 = np.nanmean(np.nanmean(tac + urban_1990_3d - urban_1990_3d, axis=0), axis=0)
        # plt.figure();
        # plt.plot(np.linspace(2001, 2023 + 11 / 12, 276), tac_urban_1990)

        # URBAN 1990-2000
        urban_1990_v2 = urban_1990 * 1
        urban_1990_v2[np.isnan(urban_1990_v2)] = 0
        urban_1990_v2[urban_1990_v2 != 0] = np.nan
        urban_1990_2000 = urban_2000 + urban_1990_v2
        urban_1990_2000_3d = np.tile(urban_1990_2000[:, :, np.newaxis], tac.shape[2])
        tac_urban_1990_2000 = np.nanmean(np.nanmean(tac + urban_1990_2000_3d - urban_1990_2000_3d, axis=0), axis=0)
        # plt.plot(np.linspace(2001, 2023 + 11 / 12, 276), tac_urban_1990_2000)

        # URBAN 2000-2018
        urban_2000_v2 = urban_2000 * 1
        urban_2000_v2[np.isnan(urban_2000_v2)] = 0
        urban_2000_v2[urban_2000_v2 != 0] = np.nan
        urban_2000_2018 = urban_2018_rsz + urban_2000_v2
        urban_2000_2018_3d = np.tile(urban_2000_2018[:, :, np.newaxis], tac.shape[2])
        tac_urban_2000_2018 = np.nanmean(np.nanmean(tac + urban_2000_2018_3d - urban_2000_2018_3d, axis=0), axis=0)
        # plt.plot(np.linspace(2001, 2023 + 11 / 12, 276), tac_urban_2000_2018)

        # URBAN-RURAL
        urban_2018_v2 = urban_2018_rsz * 1
        urban_2018_v2[np.isnan(urban_2018_v2)] = 0
        urban_2018_v2[urban_2018_v2 != 0] = np.nan
        urban_rural_near = rural_near + urban_2018_v2
        # plt.figure(); plt.imshow(rural_urban_v2)
        urban_rural_near_3d = np.tile(urban_rural_near[:, :, np.newaxis], tac.shape[2])
        tac_rural_near = np.nanmean(np.nanmean(tac + urban_rural_near_3d - urban_rural_near_3d, axis=0), axis=0)
        # plt.plot(np.linspace(2001, 2023 + 11 / 12, 276),tac_rural_near)

        # RURAL-BACKGROUND
        rural_near_v2 = rural_near * 1
        rural_near_v2[np.isnan(rural_near_v2)] = 0
        rural_near_v2[rural_near_v2 != 0] = np.nan
        urban_rural_bgr = rural_bgr + rural_near_v2
        # plt.figure(); plt.imshow(rural_urban_v2)
        urban_rural_bgr_3d = np.tile(urban_rural_bgr[:, :, np.newaxis], tac.shape[2])
        tac_rural_bgr = np.nanmean(np.nanmean(tac + urban_rural_bgr_3d - urban_rural_bgr_3d, axis=0), axis=0)
        # plt.plot(np.linspace(2001, 2023 + 11 / 12, 276),tac_rural_bgr)
        TAC.append([tac_urban_1990, tac_urban_1990_2000, tac_urban_2000_2018, tac_rural_near, tac_rural_bgr])
        print(id)
    TAC_arr = np.array(TAC)

    IDs_num = np.array([float(id) for id in IDs])
    output_file = current_dir + '/2_Output/' + 'tac_city_5zones_noBRDF' + '.npz'
    np.savez(output_file, array1=TAC_arr, array2=IDs_num)
