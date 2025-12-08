import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;

matplotlib.use('Qt5Agg')
import scipy.stats as st
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
import geopandas as gpd
from scipy.ndimage import convolve1d
import os
import rasterio

## main ##
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
coord = pd.read_csv(current_dir+'/2_Output/PICS_sites.csv')[['Lon','Lat']].values
import ee
# Authenticate to the Earth Engine servers
ee.Authenticate()
ee.Initialize()

imageC = ee.ImageCollection('LANDSAT/COMPOSITES/C02/T1_L2_32DAY_NDVI').select('NDVI').filterDate('2001-01-01', '2024-01-01').toBands()

VI_values = []
for i in range(coord.shape[0]):
    # Get the band values at the specified coordinates
    point = ee.Geometry.Point(coord[i, 0], coord[i, 1])
    band_values = imageC.reduceRegion(reducer=ee.Reducer.median(), geometry=point.buffer(100), scale=100)
    values_dict = band_values.getInfo()
    values = list(values_dict.values())
    VI_values.append(values)
    print(i)

VI_array = np.array(VI_values)
np.save(current_dir + '/2_Output/VI_values_PICS_sites.npy', VI_array)

VI_array = np.load(current_dir + '/2_Output/VI_values_PICS_sites.npy',allow_pickle=True).astype(float)

V_median = np.nanmedian(VI_array,axis=0)
V_median[V_median>0.105] = np.nan
V_median[V_median<0.057] = np.nan
# plt.figure(); plt.plot(np.linspace(2001,2023+11/12,276),V_median,'C0o')
# plt.ylim(0.04,0.12)

L5 = V_median[:12*12+3]
L8 = V_median[12*12+3:]
# plt.figure(); plt.plot(L5,'o');plt.plot(L8,'o')

bias = np.nanmean(L8) - np.nanmean(L5)
L8_2 = L8-bias
plt.figure(); plt.plot(list(L5)+list(L8_2),'o');

scale = (np.nanpercentile(L8_2,95)-np.nanpercentile(L8_2,6)) /(np.nanpercentile(L5,95)-np.nanpercentile(L5,6))
L8_3 = (L8_2-L8_2.mean())/scale+L8_2.mean()
plt.figure(); plt.plot(np.linspace(2001,2013+5/12,12*12+5),L5,'o');
plt.plot(np.linspace(2013+5/12,2023+11/12,127),L8_2,'o')
#
ser = V_median*1
EVI_yr = np.zeros_like(ser)

yr_num = 23
bands_year = 12
for year in range(yr_num):
    st = year * bands_year
    ed = st + bands_year
    evi_year = np.nanmean(ser[st:ed])
    evi_year_rep = [evi_year]*int(bands_year)
    EVI_yr[st:ed] = evi_year_rep
rm_offline = ser - EVI_yr

plt.figure(); plt.plot(np.linspace(2001,2023+11/12,276),V_median,'C0o')
plt.ylim(0.05,0.11)
plt.figure(); plt.plot(np.linspace(2001,2023+11/12,276),rm_offline+np.nanmean(EVI_yr),'C0o')
plt.ylim(0.05,0.11)