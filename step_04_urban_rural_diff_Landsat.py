import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
# import matplotlib;
# matplotlib.use('Qt5Agg')
import scipy.signal as ss
import multiprocess as mp
import os
import warnings
import cv2
warnings.filterwarnings("ignore")
import numpy as np


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    tac_folder = current_dir + '/2_Output/tac_Landsat/'
    urban_folder = current_dir + '/2_Output/nanFrac_urban_label_Landsat/'

    filenames = os.listdir(tac_folder)
    names = []
    for name in filenames:
        if name.startswith('t') and name.endswith('_variance.npy'): #'.npy'
            names.append(name)
    IDs = []
    for name in names:
        id = name.split('_')[1]#.split('npy')[0]
        IDs.append(float(id))
    IDs = np.sort(IDs)

    TAC = []
    # TAC_TMP = []
    IDs_num = []
    for id in IDs:
        tac = np.load(tac_folder + 'tac_' + str(id) + '.npy')
        tac[tac<=0]=np.nan
        tac_mean_spa = np.nanmean(tac,axis=1)

        # plt.figure(); plt.plot(np.nanmean(tac,axis=0))
        urban_nonfrac_labels = np.load(urban_folder+'label_'+str(id)+'.npy')
        urban_lab = urban_nonfrac_labels[0,:]
        nan_frac = urban_nonfrac_labels[1,:]
        # plt.close()
        # plt.figure(); plt.hist(nan_frac,50)
        # tac_inner = tac_mean_spa[urban_lab == 2].mean()
        # tac_sub = tac_mean_spa[urban_lab == 1].mean()
        # tac_rural = tac_mean_spa[urban_lab == 0].mean()
        # TAC.append([tac_inner, tac_sub, tac_rural])

        tac_tmp_inner = np.nanmedian(tac[urban_lab == 2,:],axis=0)
        tac_tmp_sub = np.nanmedian(tac[urban_lab == 1, :], axis=0)
        tac_tmp_rural = np.nanmedian(tac[urban_lab == 0, :], axis=0)
        # plt.figure(); plt.plot(tac_tmp_inner); plt.plot(tac_tmp_sub); plt.plot(tac_tmp_rural)
        tac_tmp = np.array([tac_tmp_inner, tac_tmp_sub, tac_tmp_rural])
        TAC.append(tac_tmp)


        IDs_num.append(float(id))
        print(id)
    TAC_arr = np.array(TAC)
    # TAC_TMP_arr = np.array(TAC_TMP)
    # np.nanmedian(TAC_arr,axis=0)

    tac_tmp_3z = np.nanmean(TAC_arr, axis=0)
    plt.figure(); plt.plot(tac_tmp_3z[0,:]); plt.plot(tac_tmp_3z[1,:]); plt.plot(tac_tmp_3z[2,:])

    IDs_num = np.array(IDs_num)
    output_file = current_dir + '/2_Output/' + 'tac_landsat_city_3zones' + '.npz'
    np.savez(output_file, array1=TAC_arr, array2=IDs_num)


