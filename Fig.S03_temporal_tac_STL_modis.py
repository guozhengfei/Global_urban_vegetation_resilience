import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
import matplotlib;
matplotlib.use('Qt5Agg')
import os
import numpy as np
import scipy.stats as st

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
TACs = np.load(current_dir+'/2_Output/tac_nadir_city_3zones_STL.npz')['array1'] # v5,v4.2
ID = np.load(current_dir+'/2_Output/tac_nadir_city_3zones_STL.npz')['array2'] # v5,v4.2

urban_to_rural_slopes = [] # temporal trend

for i in range(TACs.shape[0]):
    tacs = TACs[i,:,:]
    slopes = []
    for n in range(tacs.shape[0]):
        slope_i = st.linregress(np.linspace(0,tacs.shape[1]-1,tacs.shape[1])/12,tacs[n,:]).slope
        slopes.append(slope_i)
    urban_to_rural_slopes.append(slopes)

urban_to_rural_slopes = np.array(urban_to_rural_slopes)

plt.figure(); plt.hist(urban_to_rural_slopes[:,0], bins=50, ec='k',alpha=0.5)
plt.hist(urban_to_rural_slopes[:,-1], bins=50, ec='k',alpha=0.5)
plt.close()

df_tac = pd.DataFrame(urban_to_rural_slopes)
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

cities.plot('urban_core',ax=ax[0], marker='o',  markersize=5, cmap='RdBu_r',vmin=-0.1,vmax=0.1)
ax[0].set_ylim([-54,70])
ax[0].set_xlim([-170,170])


cities.plot('urban_edge',ax=ax[1], marker='o',  markersize=5, cmap='RdBu_r',vmin=-0.1,vmax=0.1)
ax[1].set_ylim([-54,70])
ax[1].set_xlim([-170,170])


cities.plot('rural_bgr',ax=ax[2], marker='o',  markersize=5, cmap='RdBu_r',vmin=-0.1,vmax=0.1)
ax[2].set_ylim([-54,70])
ax[2].set_xlim([-170,170])

tac_global_mean = np.nanmean(urban_to_rural_slopes, axis=0)

ax[3].bar(np.linspace(0,2,3),tac_global_mean,yerr=np.nanstd(urban_to_rural_slopes,axis=0)*0.1,width=0.2,color=['#7fc97f', '#beaed4', '#fdc086'])
ax[3].set_ylim([-0.015,0.015])
ax[3].set_xticks([0,1,2],['urban_core','urban_edge','rural_bgr'])

# figToPath = current_dir + '/4_Figures/Fig01b_tac_3zones'
# fig.tight_layout()
# fig.savefig(figToPath, dpi=900)
# plt.close(fig)