import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;

matplotlib.use('Qt5Agg')
import scipy.signal as ss
import multiprocess as mp
import os
import warnings

warnings.filterwarnings("ignore")
import cv2

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/nadir_ndvi_monthly_500/'
    grass_folder = current_dir + '/1_Input/grassCover_250m/'
    tree_folder = current_dir + '/1_Input/treeCover_250m/'
    crop_folder = current_dir + '/1_Input/cropCover_250m/'
    lcc_folder = current_dir + '/1_Input/land_cover_change_250m/'
    # urban_folder = current_dir+'/1_Input/urban_area/'
    filenames = os.listdir(EVI_folder)[1:]

    IDs = []
    for name in filenames:
        id = name.split('_')[-1].split('.tif')[0]
        IDs.append(id)

    tacs = []
    for id in IDs:
        EVI0 = tf.imread(EVI_folder + 'nadir_ndvi_monthly_' + id + '.tif')[:, :, :264]
        crop_frc = tf.imread(crop_folder + 'cropC_' + id + '.tif')
        crop_frc = cv2.resize(crop_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
        np.save(current_dir + '/1_Input/cropCover_250m/cropC_' + id +'.npy', crop_frc)

        grass_frc = tf.imread(grass_folder + 'grassC_' + id + '.tif')
        grass_frc = cv2.resize(grass_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
        np.save(current_dir + '/1_Input/grassCover_250m/grassC_' + id + '.npy', grass_frc)

        tree_frc = tf.imread(tree_folder + 'treeC_' + id + '.tif')
        tree_frc = cv2.resize(tree_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
        np.save(current_dir + '/1_Input/treeCover_250m/treeC_' + id + '.npy', tree_frc)

        lcc_frac = tf.imread(lcc_folder + 'lcc_' + id + '.tif')
        lcc_frac = cv2.resize(lcc_frac, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
        np.save(current_dir + '/1_Input/land_cover_change_500m/lcc_' + id + '.npy', lcc_frac)
        print(id)




