import numpy as np
import pandas as pd
import tifffile as tf
# import matplotlib.pyplot as plt
# import matplotlib;
# matplotlib.use('Qt5Agg')
import warnings
warnings.filterwarnings("ignore")
from scipy.ndimage import convolve1d
import os
import rasterio

import ee
ee.Authenticate()
ee.Initialize()

imageC = ee.ImageCollection('LANDSAT/COMPOSITES/C02/T1_L2_32DAY_NDVI').select('NDVI').filterDate('2000-01-01', '2024-01-01').toBands()


def smooth_2d_array(arr, window_size):
    kernel = np.ones(window_size) / window_size
    smoothed_arr = convolve1d(arr, kernel, axis=0, mode='nearest')
    return smoothed_arr

## main ##
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
folder = current_dir + '/1_Input/valid_area_landsat/'
# urban_folder = current_dir+'/1_Input/urban_area/'
filenames = os.listdir(folder)

names = []
for name in filenames:
    if name.startswith('v'):
        names.append(name)

IDs = []
for name in names:
    id = float(name.split('_')[-1].split('.tif')[0])
    IDs.append(id)

IDs = np.sort(IDs)
for id in IDs[347:]:
    img = tf.imread(folder + 'valid_area_' + str(id) + '.tif')
    #plt.figure(); plt.imshow(img)
    selected_indices = np.argwhere(img == 1)

    # Randomly extract 1000 samples
    np.random.shuffle(selected_indices)
    selected_indices = selected_indices[:1000]

    # plt.figure(); plt.imshow(img); plt.plot(selected_indices[:,1],selected_indices[:,0],'r.')

    src = rasterio.open(folder + 'valid_area_' + str(id) + '.tif')
    # return the coordinations of end-members
    x, y = src.xy(selected_indices[:,0], selected_indices[:,1])
    coord = np.array((x, y)).T
    lons = np.linspace(np.min(x), np.max(x),5)
    lats = np.linspace(np.min(y), np.max(y), 5)

    # np.save(current_dir+'/2_Output/Landsat_coords/sample_'+str(id)+'_coord.npy',coord)

    VI_values = []
    for i_lon in range(lons.shape[0]-1):
        lon_mask = (x >= lons[i_lon]) & (x < lons[int(i_lon+1)])
        for i_lat in range(lats.shape[0]-1):
            # Get the band values at the specified coordinates
            lat_mask = (y >= lats[i_lon]) & (y < lats[int(i_lat+1)])
            mask = lon_mask & lat_mask
            if mask.sum()==0: continue
            coord_i = coord[mask]
            features = [ee.Feature(ee.Geometry.Point(lon, lat)) for lon, lat in coord_i]
            fc = ee.FeatureCollection(features)

            band_values = imageC.reduceRegions(fc, reducer=ee.Reducer.mean(),scale=50).getInfo()

            values = list(band_values.values())
            for val in values[2]:
                vi = list(val.get('properties').values())
                x_y = list(val.get('geometry').values())[1]
                VI_values.append(vi+x_y)
        print(str(id)+'_'+str(i_lon))

    VI_array = np.array(VI_values)
    # np.save(current_dir + '/2_Output/VI_Landsat/NDVI_8d_' + str(id)+'.npy', VI_array)

    names = list(val.get('properties').keys())
    names2 = names + ['lon', 'lat']

    df_VI = pd.DataFrame(VI_array)
    df_VI.columns = names2
    df_VI.to_csv(current_dir + '/2_Output/VI_Landsat/NDVI_8d_' + str(id)+'.csv', index=False)

