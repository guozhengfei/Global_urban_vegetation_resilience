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
ee.Initialize(project='ee-zhengfei')

imageC = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').select('tmmx').filterDate('2001-01-01', '2023-01-01').toBands()

## main ##
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
folder = current_dir+'/1_Input/urban/urban_1990_250m/'
filenames = os.listdir(folder)

names = []
for name in filenames:
    if name.startswith('u'):
        names.append(name)

IDs = []
for name in names:
    id = float(name.split('_')[-1].split('.tif')[0])
    IDs.append(id)

IDs = np.sort(IDs)

# Initialize list to store all values
all_VI_values = []
column_names = None  # Initialize column names variable

for id in IDs:
    # Read the image
    img = tf.imread(folder + 'urban_1990_' + str(id) + '.tif')
    
    # Find central pixel coordinates
    center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
    
    # Open the image with rasterio to get geographic coordinates
    with rasterio.open(folder + 'urban_1990_' + str(id) + '.tif') as src:
        # Convert pixel coordinates to geographic coordinates
        center_lon, center_lat = src.xy(center_y, center_x)
        
        # Create a feature collection with single central point
        center_feature = ee.Feature(
            ee.Geometry.Point(center_lon, center_lat))
        fc = ee.FeatureCollection([center_feature])
        
        # Extract band values for central point
        band_values = imageC.reduceRegions(
            fc, 
            reducer=ee.Reducer.mean(),
            scale=5000
        ).getInfo()
        
        # Extract values and coordinates
        values = list(band_values.values())
        if len(values) > 2 and values[2]:  # Check if values exist
            val = values[2][0]  # Get first (and only) feature
            vi = list(val.get('properties').values())
            x_y = list(val.get('geometry').values())[1]
            
            # Add ID to the values
            row_values = [id] + vi + x_y
            all_VI_values.append(row_values)
            
            # Get column names from the first successful iteration
            if column_names is None:
                names = list(val.get('properties').keys())
                column_names = ['ID'] + names + ['lon', 'lat']
            
            print(f"Processing ID: {id}, Center coordinates: ({center_lon:.4f}, {center_lat:.4f})")

# Check if we have any data
if not all_VI_values:
    raise ValueError("No data was collected. Check if the input files exist and contain valid data.")

# Convert all values to array
all_VI_array = np.array(all_VI_values)

# Create and save DataFrame with all values
df_VI_all = pd.DataFrame(all_VI_array, columns=column_names)

# Save to CSV
output_path = os.path.join(current_dir, '2_Output', 'tmmx_timeseries_all.csv')
df_VI_all.to_csv(output_path, index=False)

print(f"Saved all values to: {output_path}")
