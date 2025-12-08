import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
import matplotlib; matplotlib.use('Qt5Agg')
import os
import numpy as np

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
TACs = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array1'] # v5,v4.2
ID = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array2'] # v5,v4.2

TACs_mean = np.nanmean(TACs, axis=2)
tac_global_mean = np.nanmean(TACs_mean, axis=0)

df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_edge','rural_bgr']
df_tac['ID'] = ID

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate['MAT'] = climate['MAT']-273.15
climate['MAP'] = climate['MAP']*24*1000

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
tropical = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_tropical.shp')
arid = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_arid.shp')
temperate = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_temperate.shp')
boreal = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_boreal.shp')
cities = gpd.read_file(current_dir + '/1_Input/shps/Shp/points_citis.shp')

cities = pd.merge(cities, df_tac, on='ID')
cities = pd.merge(cities, climate, on='ID')

fig, ax = plt.subplots(2,2,figsize=(6,4.5),gridspec_kw={'width_ratios': [3.5, 1]})

tropical.plot(ax=ax[0,0], color='#F78A5D', edgecolor='none',alpha=0.4)
temperate.plot(ax=ax[0,0], color='#AAD664', edgecolor='none',alpha=0.4)
arid.plot(ax=ax[0,0], color='#FFC96E', edgecolor='none',alpha=0.4)
boreal.plot(ax=ax[0,0], color='lightgrey', edgecolor='none',alpha=0.4)

# tropical.plot(ax=ax[1,0], color='#F78A5D', edgecolor='none',alpha=0.4)
# temperate.plot(ax=ax[1,0], color='#AAD664', edgecolor='none',alpha=0.4)
# arid.plot(ax=ax[1,0], color='#FFC96E', edgecolor='none',alpha=0.4)
# boreal.plot(ax=ax[1,0], color='lightgrey', edgecolor='none',alpha=0.4)

cities['core_rural_diff'] = cities['urban_core']-cities['rural_bgr']
cities1 = cities[cities['core_rural_diff']<=0]
cities1.plot('core_rural_diff', ax=ax[0,0], marker='o',  markersize=10, cmap='Blues_r', vmin=-0.2, vmax=0.1)
cities2 = cities[cities['core_rural_diff']>0]
cities2.plot('core_rural_diff',ax=ax[0,0], marker='o',  markersize=10, cmap='Reds',vmin=-0.05,vmax=0.1)
ax[0,0].set_ylim([-54, 70])
ax[0,0].set_xlim([-170, 170])
ax[0,1].scatter(cities1['MAP'], cities1['MAT'], s=12, c=cities1['core_rural_diff'], cmap='Blues_r', vmin=-0.2, vmax=0.1)
ax[0,1].scatter(cities2['MAP'], cities2['MAT'], s=12, c=cities2['core_rural_diff'], cmap='Reds', vmin=-0.05, vmax=0.1)

# cities['edge_rural_diff'] = cities['urban_edge']-cities['rural_bgr']
# cities1 = cities[cities['edge_rural_diff']<=0]
# cities1.plot('edge_rural_diff',ax=ax[1,0], marker='o',  markersize=10, cmap='Blues_r',vmin=-0.2,vmax=0.1)
# cities2 = cities[cities['edge_rural_diff']>0]
# cities2.plot('edge_rural_diff',ax=ax[1,0], marker='o',  markersize=10, cmap='Reds',vmin=-0.05,vmax=0.1)
# ax[1,0].set_ylim([-54,70])
# ax[1,0].set_xlim([-170,170])
# ax[1,1].scatter(cities1['MAP'], cities1['MAT'], s=12, c=cities1['core_rural_diff'], cmap='Blues_r', vmin=-0.2, vmax=0.1)
# ax[1,1].scatter(cities2['MAP'], cities2['MAT'], s=12, c=cities2['core_rural_diff'], cmap='Reds', vmin=-0.05, vmax=0.1)

figToPath = current_dir + '/4_Figures/Fig01c_urban_rural_diff'
fig.tight_layout()
fig.savefig(figToPath, dpi=900)
# plt.close(fig)

fig, ax = plt.subplots(1,2,figsize=(3, 1.5))
ax[0].hist(cities['core_rural_diff'],bins=20,ec='k')
ax[1].hist(cities['edge_rural_diff'],bins=20,ec='k')
figToPath = current_dir + '/4_Figures/Fig01c_urban_rural_diff_hist'
fig.tight_layout()
# fig.savefig(figToPath, dpi=600)
# plt.close(fig)

import scipy.stats as st
plt.figure(); plt.plot(cities['MAP'], cities['core_rural_diff'],'o',mfc='none')
st.linregress(cities['MAP'], cities['core_rural_diff'])
plt.figure(); plt.plot(cities['MAT'], cities['core_rural_diff'],'o',mfc='none')
st.linregress(cities['MAT'], cities['core_rural_diff'])

#urban examples with higher resilience

fig, ax = plt.subplots(1,2,figsize=(10, 5))

tropical.plot(ax=ax[0], color='#F78A5D', edgecolor='none',alpha=0.4)
temperate.plot(ax=ax[0], color='#AAD664', edgecolor='none',alpha=0.4)
arid.plot(ax=ax[0], color='#FFC96E', edgecolor='none',alpha=0.4)
boreal.plot(ax=ax[0], color='lightgrey', edgecolor='none',alpha=0.4)

tropical.plot(ax=ax[1], color='#F78A5D', edgecolor='none',alpha=0.4)
temperate.plot(ax=ax[1], color='#AAD664', edgecolor='none',alpha=0.4)
arid.plot(ax=ax[1], color='#FFC96E', edgecolor='none',alpha=0.4)
boreal.plot(ax=ax[1], color='lightgrey', edgecolor='none',alpha=0.4)

cities['core_rural_diff'] = cities['urban_core']-cities['rural_bgr']
cities1 = cities[cities['core_rural_diff']<=0]
cities1.plot('core_rural_diff', ax=ax[0], marker='o',  markersize=10, cmap='Blues_r', vmin=-0.2, vmax=0.1)
cities2 = cities[cities['core_rural_diff']>0]
cities2.plot('core_rural_diff',ax=ax[0], marker='o',  markersize=10, cmap='Reds',vmin=-0.05,vmax=0.1)
# ax[0,0].set_ylim([-54, 70])
# ax[0,0].set_xlim([-170, 170])

plt.show()