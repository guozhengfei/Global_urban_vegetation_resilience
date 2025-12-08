import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import os
import rasterio
import ee

ee.Authenticate()
ee.Initialize(project='ee-zhengfei')

points_df_dir = os.path.join('..', '2_Output', 'pure_veg', 'urban_factors_effect_v2.csv')
points = pd.read_csv(points_df_dir)
coords = list(zip(points['lon'], points['lat']))

geotiff_path = os.path.join('..', '2_Output', 'openET_clipped_image.tif')
src = rasterio.open(geotiff_path)
values = []
valid_points = []
valid_indices = []

for i, (lon, lat) in enumerate(coords):
    try:
        row, col = src.index(lon, lat)
        if 0 <= row < src.height and 0 <= col < src.width:
            band_value = src.read(1, window=((row, row + 1), (col, col + 1))).flatten()[0]
            if band_value > 0:
                values.append(band_value)
                valid_points.append((lon, lat))
                valid_indices.append(i)
    except Exception as e:
        continue  # skip points with errors

start_date = '2001-01-01'
end_date = '2023-12-31'

# OpenET (monthly)
et_collection = ee.ImageCollection('OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0').select('et_ensemble_mad')

def extract_monthly_series(lon, lat, collection, start_date, end_date):
    point = ee.Geometry.Point([lon, lat])
    filtered = collection.filterDate(start_date, end_date).filterBounds(point)
    band_names = filtered.aggregate_array('system:index').getInfo()
    img_bands = filtered.toBands()
    values = img_bands.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=30,
        maxPixels=1e9
    ).getInfo()
    # Ensure order matches band_names
    result = [values.get(band, np.nan) for band in values.keys()]
    return result, band_names

et_series = []
band_names = None
for i, (lon, lat) in enumerate(valid_points):
    try:
        vals, bands = extract_monthly_series(lon, lat, et_collection, start_date, end_date)
        if band_names is None:
            band_names = bands
        et_series.append([lon, lat] + vals)
        print(f'Processed point {i+1}/{len(valid_points)}')
    except Exception as e:
        print(f'Error at point {i+1}: {e}')
        continue

et_series2=[]
for data in et_series:
    if len(data)!=278:
        data0 = np.array(data[2:276+2]).astype(float)
        data0_nan_count = np.isnan(data0).sum()
        data1 = np.array(data[276+2:]).astype(float)
        data1_nan_count = np.isnan(data1).sum()
        if data0_nan_count<data1_nan_count:
            et_series2.append(data[0:2]+list(data0))
        else:
            et_series2.append(data[0:2]+list(data1))
    else:
        et_series2.append(data)

band_names_new = []
for name in band_names:
    band_names_new.append(name.split('_')[1])

# Prepare DataFrame

columns = ['lon', 'lat'] + band_names_new
et_df = pd.DataFrame(et_series2, columns=columns)

valid_df = points.iloc[valid_indices,:].set_index(et_df.index)
et_df = pd.concat([valid_df,et_df.iloc[:,2:]],axis=1)
output_path = os.path.join('..', '2_Output', 'pure_veg', 'monthly_openET_2001_2023.csv')
et_df.to_csv(output_path, index=False)