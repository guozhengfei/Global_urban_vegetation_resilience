import geopandas as gpd
import geodatasets
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib#; matplotlib.use('Qt5Agg')
import os
import numpy as np
import seaborn as sns

plt.rc('font', family='Arial')
plt.rc('lines', linewidth=1.0)  # Increase default line width
plt.rc('axes', linewidth=1.0)   # Increase axes line width
plt.rc('grid', linewidth=1.0)   # Increase grid line width
plt.tick_params(width=1.0, labelsize=14)
plt.rc('xtick.major', size=4, width=1.0)    # Increase length and width of major ticks
plt.rc('ytick.major', size=4, width=1.0)    # Increase length and width of major ticks
plt.close()

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
TACs = np.load(current_dir+'/2_Output/tac_landsat_city_3zones.npz')['array1'] #
ID = np.load(current_dir+'/2_Output/tac_landsat_city_3zones.npz')['array2'] #
ID2 = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array2'] #

TACs_mean = np.nanmean(TACs, axis=2)
tac_global_mean = np.nanmean(TACs_mean, axis=0)

df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_edge','rural_bgr']
df_tac['ID'] = ID
df_tac = df_tac[df_tac['ID'].isin(ID2)]

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate['MAT'] = climate['MAT']-273.15
climate['MAP'] = climate['MAP']*24*1000

coastline = gpd.read_file(geodatasets.get_path('naturalearth.land'))
tropical = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_tropical.shp')
arid = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_arid.shp')
temperate = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_temperate.shp')
boreal = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_boreal.shp')
cities = gpd.read_file(current_dir + '/1_Input/shps/Shp/points_citis.shp')

cities = pd.merge(cities, df_tac, on='ID')
cities = pd.merge(cities, climate, on='ID')

fig, ax = plt.subplots(1, 3, figsize=(10.5*0.8, 2.9*0.8), gridspec_kw={'width_ratios': [2, 0.5, 1]})
coastline.boundary.plot(ax=ax[0], color='k', linewidth=0.5, zorder=10)

# Plot climate zones and cities with core_rural_diff
tropical.plot(ax=ax[0], color='#F78A5D', edgecolor='none', alpha=0.4)
temperate.plot(ax=ax[0], color='#AAD664', edgecolor='none', alpha=0.4)
arid.plot(ax=ax[0], color='#FFC96E', edgecolor='none', alpha=0.4)
boreal.plot(ax=ax[0], color='lightgrey', edgecolor='none', alpha=0.4)

cities['core_rural_diff'] = cities['urban_core']-cities['rural_bgr']
cities.plot('core_rural_diff', ax=ax[0], marker='o', markersize=10, cmap='BrBG_r', vmin=-0.1, vmax=0.1)

ax[0].set_ylim([-54, 80])
ax[0].set_xlim([-149, 170])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel('')
ax[0].set_ylabel('')
# Calculate mean core_rural_diff for each 5-degree latitude bin
cities['latitude_bin'] = (cities.geometry.y // 5) * 5
latitude_bins = cities.groupby('latitude_bin')['core_rural_diff'].agg(['mean', 'std']).reset_index()

# Remove the bin values at -40 latitude
latitude_bins = latitude_bins[latitude_bins['latitude_bin'] != -40]

# Rotate the second panel 90 degrees and plot
ax[1].plot(latitude_bins['mean'], latitude_bins['latitude_bin'], color='k',lw=1.5)
ax[1].fill_betweenx(latitude_bins['latitude_bin'], latitude_bins['mean'] - 0.5 * latitude_bins['std'], latitude_bins['mean'] + 0.5 * latitude_bins['std'], color='k', alpha=0.2)
ax[1].invert_yaxis()  # Invert y-axis to match the map
ax[1].set_ylim([-50, 70])
ax[1].set_xlim([-0.14, 0.04])

# Create a heatmap for core_rural_diff distribution in MAT and MAP space with 9x9 grid
cities['MAT_bin'] = pd.cut(cities['MAT'], bins=9)
cities['MAP_bin'] = pd.cut(cities['MAP'], bins=9)
heatmap_data = cities.pivot_table(index='MAT_bin', columns='MAP_bin', values='core_rural_diff', aggfunc='mean')

# Prepare data for pcolor
heatmap_data = heatmap_data.values
x_edges = np.linspace(cities['MAP'].min(), cities['MAP'].max(), heatmap_data.shape[1] + 1)
y_edges = np.linspace(cities['MAT'].min(), cities['MAT'].max(), heatmap_data.shape[0] + 1)

# Plot using pcolor
c = ax[2].pcolor(x_edges, y_edges, heatmap_data, cmap='BrBG_r', edgecolors='k', linewidths=1, vmin=-0.1, vmax=0.1)
fig.colorbar(c, ax=ax[2], label='Core-Rural Difference')

ax[2].set_xlabel('MAP (mm/year)')
ax[2].set_ylabel('MAT (°C)')
ax[2].tick_params(left=False, bottom=False)  # Hide the ticks and labels

figToPath = current_dir + '/4_Figures/Fig03_urban_rural_diff_landsat'
fig.tight_layout()
fig.savefig(figToPath, dpi=900)
# plt.close(fig)

plt.figure(figsize=(1.4, 1.4))
plt.hist(cities['core_rural_diff'], 20, [-0.13,0.13],color='g', alpha=0.7,edgecolor='k')
plt.xlabel('ΔTAC')
plt.tight_layout()
figToPath = current_dir + '/4_Figures/Fig03_urban_rural_diff_landsat_hist'
plt.savefig(figToPath, dpi=900)