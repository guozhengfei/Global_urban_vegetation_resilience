## 1. using openET to test  our method ET to estimate ET using LST
import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;

matplotlib.use('Qt5Agg')
import os
import rasterio
import scipy.stats as st
import seaborn as sns



df_fluxnet_dir = os.path.join('..','..', 'EC_data_halfhour_v3.csv')
df_fluxnet = pd.read_csv(df_fluxnet_dir)
# et = df_et.iloc[:,1:-4].values

etref_df_dir = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'ETref_era5L_csv_urban_751.csv')
etref_df = pd.read_csv(etref_df_dir)

openet_path = os.path.join('..', '2_Output', 'pure_veg', 'monthly_openET_2001_2023.csv')
openet_df = pd.read_csv(openet_path)

rn_path1 = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'Rn_lw_era5L_csv_urban_751.csv')
rn_df1 = pd.read_csv(rn_path1)

rn_path2 = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'Rn_sw_era5L_csv_urban_751.csv')
rn_df2 = pd.read_csv(rn_path2)

ta_path = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'Ta_era5L_csv_urban_751.csv')
ta_df = pd.read_csv(ta_path)

td_path = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'dewpoint_temperature_era5L_csv_urban_751.csv')
td_df = pd.read_csv(td_path)

lst_path = os.path.join('..', '2_Output', 'pure_veg', 'monthly_landsat_lst.csv')
lst_df = pd.read_csv(lst_path)
IDs = np.sort(list(set(openet_df['id'])))

plt.show()