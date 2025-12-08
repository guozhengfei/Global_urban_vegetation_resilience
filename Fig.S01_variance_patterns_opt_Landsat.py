import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import geodatasets
import os
import numpy as np
import seaborn as sns

plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
plt.close()

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
relative_path = '/2_Output/variance_landsat_city_3zones.npz'
relative_path2 = '/2_Output/variance_nadir_city_3zones.npz'

TACs = np.load(current_dir+relative_path)['array1'] # v5,v4.2
ID = np.load(current_dir+relative_path)['array2'] # v5,v4.2
ID2 = np.load(current_dir+relative_path2)['array2']

TACs_mean = np.nanmean(TACs, axis=2)
tac_global_mean = np.nanmean(TACs_mean, axis=0)
plt.figure(); plt.hist(TACs_mean.reshape(-1),50)

fig, ax = plt.subplots(2, figsize=(3.5, 6.5))
ax[0].plot(TACs_mean[:,0],TACs_mean[:,1],'o',mfc='none')
ax[1].plot(TACs_mean[:,0],TACs_mean[:,2],'o',mfc='none')
plt.close()

df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_edge','rural_bgr']
df_tac['ID'] = ID
df_tac = df_tac[df_tac['ID'].isin(ID2)]

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate['MAT'] = climate['MAT']-273.15
climate['MAP'] = climate['MAP']*24*1000

world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
tropical = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_tropical.shp')
arid = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_arid.shp')
temperate = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_temperate.shp')
boreal = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_boreal.shp')
cities = gpd.read_file(current_dir + '/1_Input/shps/Shp/points_citis.shp')

cities = pd.merge(cities, df_tac, on='ID')
cities = pd.merge(cities, climate, on='ID')

fig, ax = plt.subplots(3,1,figsize=(5*0.9,6*0.9))

tropical.plot(ax=ax[0], color='#F78A5D', edgecolor='none',alpha=0.4)
temperate.plot(ax=ax[0], color='#AAD664', edgecolor='none',alpha=0.4)
arid.plot(ax=ax[0], color='#FFC96E', edgecolor='none',alpha=0.4)
boreal.plot(ax=ax[0], color='lightgrey', edgecolor='none',alpha=0.4)

tropical.plot(ax=ax[1], color='#F78A5D', edgecolor='none',alpha=0.4)
temperate.plot(ax=ax[1], color='#AAD664', edgecolor='none',alpha=0.4)
arid.plot(ax=ax[1], color='#FFC96E', edgecolor='none',alpha=0.4)
boreal.plot(ax=ax[1], color='lightgrey', edgecolor='none',alpha=0.4)

tropical.plot(ax=ax[2], color='#F78A5D', edgecolor='none',alpha=0.4)
temperate.plot(ax=ax[2], color='#AAD664', edgecolor='none',alpha=0.4)
arid.plot(ax=ax[2], color='#FFC96E', edgecolor='none',alpha=0.4)
boreal.plot(ax=ax[2], color='lightgrey', edgecolor='none',alpha=0.4)

cities.plot('urban_core',ax=ax[0], marker='o',  markersize=5, cmap='RdBu_r',vmin=0.0003,vmax=0.0025)
ax[0].set_ylim([-54,80])
ax[0].set_xlim([-149,170])

cities.plot('urban_edge',ax=ax[1], marker='o',  markersize=5,cmap='RdBu_r',vmin=0.0003,vmax=0.0025)
ax[1].set_ylim([-54,80])
ax[1].set_xlim([-149,170])

cities.plot('rural_bgr',ax=ax[2], marker='o',  markersize=5,cmap='RdBu_r',vmin=0.0003,vmax=0.0025)
ax[2].set_ylim([-54,80])
ax[2].set_xlim([-149,170])

figToPath = current_dir + '/4_Figures/Fig01a_tac_3zones_landsat'
fig.tight_layout()
fig.savefig(figToPath, dpi=900)

# fig, ax = plt.subplots(1,1,figsize=(4.3*0.9,2.1*0.9))
# ax.bar(np.linspace(0,2,3),tac_global_mean,yerr=np.nanstd(TACs_mean,axis=0)*0.25,width=0.45,color=['#7fc97f', '#beaed4', '#fdc086'])
# ax.set_ylim([0.0006,0.0021])
# ax.set_xticks([0,1,2],['urban_core','urban_edge','rural_bgr'])
# ax.set_yticks([0.0009,0.0012,0.0015,0.0018])
# figToPath = current_dir + '/4_Figures/Fig01a_variance_3zones_landsat_bar'
# fig.tight_layout()
# fig.savefig(figToPath, dpi=900)
#
# fig, ax = plt.subplots(1,3,figsize=(4*0.6,1.8*0.6),sharex=True,sharey=True)
# sns.histplot(cities['urban_core'], bins=10, ax=ax[0], color='C0')
# sns.histplot(cities['urban_edge'], bins=10,  ax=ax[1], color='C0')
# sns.histplot(cities['rural_bgr'], bins=10,  ax=ax[2], color='C0')
#
# figToPath = current_dir + '/4_Figures/Fig01a_variance_3zones_landsat_hist'
# fig.tight_layout()
# fig.savefig(figToPath, dpi=900)