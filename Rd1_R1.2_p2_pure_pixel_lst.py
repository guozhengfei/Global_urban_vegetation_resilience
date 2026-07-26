## extract the landsat LST for pure veg area in urban and rural
import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import os
import rasterio

points_df_dir = os.path.join('..', '2_Output', 'pure_veg', 'urban_factors_effect_100.csv')
points = pd.read_csv(points_df_dir)

lst_dir_path = os.path.join('/Volumes/Zhengfei_02/Zhengfei_01-BACKUP', 'project 1 urban vegetation thermoregulation','1_Input', 'LST')
lst_filenames = os.listdir(lst_dir_path)

IDs = []
for name in lst_filenames:
    if name.startswith('Landsat'):
        IDs.append(name.split('_')[1].split('.tif')[0])
IDs = sorted(np.array(IDs).astype(float))[1:]

lst_df = points.copy()
lst_df[['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']]=np.nan
for id in IDs:
    lst_path_i = lst_dir_path + '/Landsat_' + str(id) + '.tif'
    src = rasterio.open(lst_path_i)
    coord_i = points.loc[points['id'] == id]
    lons = coord_i['lon'].values
    lats = coord_i['lat'].values
    data = src.read()

    city_values = []
    for i in range(len(lons)):
        row, col = src.index(lons[i], lats[i])
        band_values = data[:, row, col]
        city_values.append(band_values)
    city_df = pd.DataFrame(city_values, index=coord_i.index)
    # check if city_df is 12 columns, if not, add a column using the mean of first and last column, until equal to 12 columns
    while city_df.shape[1] < 12:
        mean_col = city_df.apply(lambda row: np.nanmean([row.iloc[0], row.iloc[-1]]), axis=1)
        city_df[city_df.shape[1]] = mean_col
    # If more than 12 columns, trim to 12
    if city_df.shape[1] > 12:
        city_df = city_df.iloc[:, :12]
    # Rename columns to m1...m12
    city_df.columns = ['m'+str(i+1) for i in range(12)]

    # Ensure the assignment matches the indices
    lst_df.loc[coord_i.index, ['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']] = city_df.values
    print(id)

output_path = os.path.join('..', '2_Output', 'pure_veg', 'monthly_landsat_lst_100.csv')
lst_df.to_csv(output_path, index=False)