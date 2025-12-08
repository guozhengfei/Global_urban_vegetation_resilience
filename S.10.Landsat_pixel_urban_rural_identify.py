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
import rasterio


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    folder = current_dir + '/2_Output/VI_Landsat/'
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    tac_folder = current_dir + '/2_Output/tac_500m_yr/'
    urban_folder_2018 = current_dir + '/1_Input/urban/urban_2018_1000m/'
    urban_folder_1990 = current_dir + '/1_Input/urban/urban_1990_250m/'

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
        coords = df_i.iloc[:,-2:].values
        lats = coords[:, 1]
        lons = coords[:, 0]

        src = rasterio.open(urban_folder_2018+'fvc_'+str(id)+'.tif')
        rows, cols = np.array(src.index(lons, lats))
        data = src.read(1)
        rows[rows >= data.shape[0]] = int(data.shape[0] - 1)
        cols[cols >= data.shape[1]] = int(data.shape[1] - 1)
        value1 = data[rows, cols]
        value1[value1!=0]=1

        src = rasterio.open(urban_folder_1990+'urban_1990_'+str(id)+'.tif')
        rows, cols = np.array(src.index(lons, lats))
        data = src.read(1)
        rows[rows >= data.shape[0]] = int(data.shape[0] - 1)
        cols[cols >= data.shape[1]] = int(data.shape[1] - 1)
        value2 = data[rows, cols]
        value2[value2!=0]=1

        value3 = value1+value2 # 2: inner city; 1: sub-city; 0:rural
        # plt.figure(); plt.imshow(data); plt.plot(cols,rows,'ro')
        nan_frac = np.sum(np.isnan(vis),axis=1)/vis.shape[1]
        (value3==2).sum()

        result = np.stack([value3,nan_frac],axis=0)
        # plt.figure(); plt.hist(np.sum(np.isnan(vis),axis=1)/288,50)

        output_file = current_dir + '/2_Output/nanFrac_urban_label_Landsat/' + 'label_' + str(id) + '.npy' # first row: urban label; second row: nan fraction
        np.save(output_file, result)
        print(id)
