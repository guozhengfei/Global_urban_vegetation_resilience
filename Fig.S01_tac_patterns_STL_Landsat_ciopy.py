import geopandas as gpd
import geodatasets
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import os
import numpy as np
import seaborn as sns

plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
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

fig, ax = plt.subplots(1, 2, figsize=(10.5*0.8, 3.9*0.8), gridspec_kw={'width_ratios': [2, 0.5]})

# Plot world coastline using land boundaries
coastline.boundary.plot(ax=ax[0], color='k', linewidth=0.5, zorder=10)

# Plot climate zones and cities with core_rural_diff
tropical.plot(ax=ax[0], color='#F78A5D', edgecolor='none', alpha=0.4)
temperate.plot(ax=ax[0], color='#AAD664', edgecolor='none', alpha=0.4)
arid.plot(ax=ax[0], color='#FFC96E', edgecolor='none', alpha=0.4)
boreal.plot(ax=ax[0], color='lightgrey', edgecolor='none', alpha=0.4)

cities['core_rural_diff'] = 0.12
cities.plot('core_rural_diff', ax=ax[0], marker='o', markersize=10, cmap='RdBu_r', vmin=0.08, vmax=0.25,edgecolor='b')

ax[0].set_ylim([-54, 80])
ax[0].set_xlim([-149, 170])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel('')
ax[0].set_ylabel('')


figToPath = current_dir + '/4_Figures/cities_location'
fig.tight_layout()
fig.savefig(figToPath, dpi=900)
# plt.close(fig)
