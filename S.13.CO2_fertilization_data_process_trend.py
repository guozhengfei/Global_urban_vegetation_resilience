import os
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')

def cal_slope(y):
    # Create x array (years)
    x = np.arange(len(y))
    
    # Remove NaN values
    mask = ~np.isnan(y)
    if np.sum(mask) < 3:  # Need at least 3 points for meaningful trend
        return np.nan
    
    x_valid = x[mask]
    y_valid = y[mask]
    
    try:
        # Calculate linear regression
        A = np.vstack([x_valid, np.ones(len(x_valid))]).T
        slope, _ = np.linalg.lstsq(A, y_valid, rcond=None)[0]
        return slope
    except:
        return np.nan

base_dir = os.path.join('..', '1_Input', 'CO2')
output_file = os.path.join('..', '1_Input','CO2', 'CO2_2000_2020.nc.npy')
all_data = np.load(output_file)
mean_co2 = all_data#np.nanmean(all_data, axis=0)


# Define the directory for the urban GeoTIFF files
import rasterio
urban_tiff_dir = os.path.join('..', '1_Input', 'urban', 'urban_1990_250m')

# List of urban GeoTIFF files

urban_tiff_files = [os.path.join(urban_tiff_dir, f) for f in os.listdir(urban_tiff_dir) if f.endswith('.tif')]


# Function to extract mean CO2 for a city
def extract_mean_co2(city_tiff_file, mean_co2):
    with rasterio.open(city_tiff_file) as src:
        city_data = src.read(1)
        city_mask = city_data > 0  # Assuming non-zero values represent the city area
        city_coords = np.column_stack(np.where(city_mask))

        # Get the affine transformation of the GeoTIFF
        transform = src.transform

        # Calculate the urban core CO2 (central coordinate pixel)
        central_index = len(city_coords) // 2
        central_coord = city_coords[central_index]
        central_lon_lat = transform * (central_coord[1], central_coord[0])
        central_i, central_j = int((90 - central_lon_lat[1]) / 0.05), int((central_lon_lat[0] + 180) / 0.05)
        urban_core_co2 = mean_co2[:,central_i, central_j] if 0 <= central_i < mean_co2.shape[1] and 0 <= central_j < mean_co2.shape[2] else [np.nan]*mean_co2.shape[0]

        # Calculate the urban outedge CO2 (mean CO2 of the four vertices)
        vertices = [city_coords[0], city_coords[-1], city_coords[len(city_coords) // 4], city_coords[3 * len(city_coords) // 4]]
        vertex_lon_lat = [transform * (v[1], v[0]) for v in vertices]
        vertex_indices = [(int((90 - lat) / 0.05), int((lon + 180) / 0.05)) for lon, lat in vertex_lon_lat]
        vertex_co2_values = [mean_co2[:,i, j] for i, j in vertex_indices if 0 <= i < mean_co2.shape[1] and 0 <= j < mean_co2.shape[2]]
        urban_outedge_co2 = np.nanmean(vertex_co2_values,axis=0)

        return cal_slope(urban_core_co2), cal_slope(urban_outedge_co2)

# Extract mean CO2 for each city
city_co2_means = {}
for city_tiff_file in urban_tiff_files:
    city_name = os.path.basename(city_tiff_file).split('_')[-1].split('.tif')[0]
    urban_core_co2, urban_outedge_co2 = extract_mean_co2(city_tiff_file, mean_co2)
    city_co2_means[city_name] = {'urban_core_co2_trend': urban_core_co2, 'urban_outedge_co2_trend': urban_outedge_co2}

# Optionally, save the city CO2 means to a CSV file
import pandas as pd

city_co2_df = pd.DataFrame.from_dict(city_co2_means, orient='index').reset_index()
city_co2_df.columns = ['ID', 'Urban_Core_CO2_trend', 'Urban_Outedge_CO2_trend']
output_csv_file = os.path.join('..', '2_Output', 'city_CO2_trend.csv')
city_co2_df.to_csv(output_csv_file, index=False)
