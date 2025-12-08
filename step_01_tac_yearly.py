import numpy as np
import tifffile as tf
# import matplotlib.pyplot as plt
# import matplotlib; matplotlib.use('Qt5Agg')
import scipy.signal as ss
import multiprocess as mp
import os
import warnings

warnings.filterwarnings("ignore")
import cv2

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    tac_folder = current_dir + '/2_Output/tac_500m_yr/'
    filenames = os.listdir(tac_folder)
    names = []
    for name in filenames:
        if name.startswith('t') and name.endswith('_variance.npy'):
            names.append(name)

    for name in names:
        tac_i = np.load(tac_folder+name)
        years = int(tac_i.shape[2]/12)
        tac_yr = np.empty_like(tac_i[:,:,:years])
        for m in range(years):
            tac_yr[:,:,m] = np.nanmean(tac_i[:,:,int(m*12):int((m+1)*12)],axis=2)

        # plt.figure(); plt.plot(np.nanmean(np.nanmean(tac_yr,axis=0),axis=0))

        output_file = current_dir + '/2_Output/tac_500m_yr2/' + name
        np.save(output_file, tac_yr)
        print(name)
