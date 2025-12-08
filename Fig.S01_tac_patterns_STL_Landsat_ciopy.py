import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
plt.close()
import matplotlib;
matplotlib.use('Qt5Agg')
import os
import numpy as np

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
relative_path = '/2_Output/tac_landsat_city_3zones_STL.npz'
# relative_path = '/2_Output/tac_city_5zones_noBRDF.npz'

TACs = np.load(current_dir+relative_path)['array1'] # v5,v4.2
ID = np.load(current_dir+relative_path)['array2'] # v5,v4.2

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

fig, ax = plt.subplots(4,1,figsize=(5*0.9,8*0.9))

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

cities.plot('urban_core',ax=ax[0], marker='o',  markersize=5, cmap='RdBu_r',vmin=0.07,vmax=0.3)
ax[0].set_ylim([-54,70])
ax[0].set_xlim([-170,170])

cities.plot('urban_edge',ax=ax[1], marker='o',  markersize=5,cmap='RdBu_r',vmin=0.07,vmax=0.3)
ax[1].set_ylim([-54,70])
ax[1].set_xlim([-170,170])

cities.plot('rural_bgr',ax=ax[2], marker='o',  markersize=5,cmap='RdBu_r',vmin=0.07,vmax=0.3)
ax[2].set_ylim([-54,70])
ax[2].set_xlim([-170,170])

ax[3].bar(np.linspace(0,2,3),tac_global_mean,yerr=np.nanstd(TACs_mean,axis=0)*0.1,width=0.45,color=['#7fc97f', '#beaed4', '#fdc086'])
ax[3].set_ylim([0.14,0.2])
ax[3].set_xticks([0,1,2],['urban_core','urban_edge','rural_bgr'])

figToPath = current_dir + '/4_Figures/Fig01a_tac_3zones_landsat_STL'
fig.tight_layout()
fig.savefig(figToPath, dpi=900)
plt.close(fig)

# tac across climate zone
# plt.hist(df_tac['urban_core']-df_tac['rural_bgr'],50)
# plt.show()