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
relative_path = '/2_Output/tac_250_city_5zones.npz'
TACs = np.load(current_dir+'/2_Output/tac_nadir_city_5zones_new.npz')['array1'] # v5,v4.2
ID = np.load(current_dir+'/2_Output/tac_nadir_city_5zones_new.npz')['array2'] # v5,v4.2

# urban_to_rural_slopes = [] # temporal trend
#
# for i in range(TACs.shape[0]):
#     tacs = TACs[i,:,:]
#     slopes = []
#     for n in range(tacs.shape[0]):
#         slope_i = st.linregress(np.linspace(0,tacs.shape[1]-1,tacs.shape[1])/12,tacs[n,:]).slope
#         slopes.append(slope_i)
#     urban_to_rural_slopes.append(slopes)
#
# urban_to_rural_slopes = np.array(urban_to_rural_slopes)
#
# plt.figure(); plt.hist(urban_to_rural_slopes[:,0], bins=50, ec='k',alpha=0.5)
# plt.hist(urban_to_rural_slopes[:,4], bins=50, ec='k',alpha=0.5)
#
# df_tac = pd.DataFrame(urban_to_rural_slopes)
TACs_mean = np.nanmean(TACs, axis=2)
df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_newtown','urban_edge','rural_edge','rural_bgr']
df_tac['ID'] = ID

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')

climate['koppen'][climate['koppen'] <= 3] = 1
climate['koppen'][(climate['koppen'] > 3) & (climate['koppen'] <= 7)] = 2
climate['koppen'][(climate['koppen'] > 7) & (climate['koppen'] <= 16)] = 3
climate['koppen'][(climate['koppen'] > 16) & (climate['koppen'] <= 28)] = 4
climate['koppen'][(climate['koppen'] > 28) & (climate['koppen'] <= 30)] = 5

cities = gpd.read_file(current_dir + '/2_Output/Shp/points_citis.shp')
gdp_per = pd.read_csv(current_dir + '/2_Output/GDP_per_capita.csv')
world_countries = gpd.read_file(current_dir + '/2_Output/Shp/world_countries.shp')
world_countries = world_countries.rename(columns={'color_code':'Country Code'})
world_countries = pd.merge(world_countries,gdp_per,on='Country Code')

cities = pd.merge(cities, df_tac, on='ID')
cities = pd.merge(cities, climate, on='ID')

tac_climate_mean = []
tac_climate_sd = []
for i in [1,3,4,2]:
    tropical_city = cities[cities['koppen']==i]
    tac_tropical_mean = tropical_city.iloc[:,3:8].mean().values
    tac_climate_mean.append(tac_tropical_mean)
    tac_tropical_sd = tropical_city.iloc[:, 3:8].std().values*0.5
    tac_climate_sd.append(tac_tropical_sd)

tac_climate_mean = np.array(tac_climate_mean)
tac_climate_sd = np.array(tac_climate_sd)

colors = ['#7fc97f', '#beaed4', '#fdc086',  '#80b1d3', '#fb8072']

fig, ax = plt.subplots(1,figsize=(9,3.3))
x = np.array([0.7,0.85, 1, 1.15,1.3])
ax.bar(x, tac_climate_mean[0,:], yerr=tac_climate_sd[0,:], width=0.15,color=colors) # tropical

ax.bar(x+1, tac_climate_mean[1,:], yerr=tac_climate_sd[1,:], width=0.15,color=colors) # temperate

ax.bar(x+2, tac_climate_mean[2,:], yerr=tac_climate_sd[2,:], width=0.15,color=colors) # boreal

ax.bar(x+3, tac_climate_mean[3,:], yerr=tac_climate_sd[3,:], width=0.15,color=colors) # arid

ax.bar(x, tac_climate_mean[0,:], yerr=tac_climate_sd[0,:], width=0.15,color=colors) # tropical
ax.set_ylim([0.1,0.55])

ax.set_xticks([1,2,3,4], ['Tropical', 'Temperate', 'Boreal', 'Arid'])
ax.tick_params(labelsize=12)

figToPath = current_dir + '/4_Figures/Fig01b_tac_by_koppen'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
# plt.close(fig)


# tac_trend_mean = []
# Developing_tac_trend_mean = []
# Developed_tac_trend_sd = []
# Developing_tac_trend_sd = []
#
# for i in [1,3,4,2]:
#     tropical_city = cities[cities['koppen']==i]
#     points_in_polygons = gpd.sjoin(tropical_city, world_countries, how="left", op='intersects')
#     tropical_developed = points_in_polygons[points_in_polygons['2018']>12375]
#     tropical_developing = points_in_polygons[points_in_polygons['2018']<12375]
#
#     Developed_tac_trend_mean.append(tropical_developed.iloc[:,3:8].mean().values*0.5)
#     Developed_tac_trend_sd.append(tropical_developed.iloc[:, 3:8].std().values*0.5)
#
#     Developing_tac_trend_mean.append(tropical_developing.iloc[:,3:8].mean().values*0.5)
#     Developing_tac_trend_sd.append(tropical_developing.iloc[:, 3:8].mean().values*0.5)
#
# Developed_tac_trend_mean = np.array(Developed_tac_trend_mean)
# Developing_tac_trend_mean = np.array(Developing_tac_trend_mean)
# Developed_tac_trend_sd = np.array(Developed_tac_trend_sd)
# Developing_tac_trend_sd = np.array(Developing_tac_trend_sd)
#
#
# Developed_tac_trend_mean = []
# Developing_tac_trend_mean = []
# Developed_tac_trend_sd = []
# Developing_tac_trend_sd = []
#
# for i in [1,3,4,2]:
#     tropical_city = cities[cities['koppen']==i]
#     points_in_polygons = gpd.sjoin(tropical_city, world_countries, how="left", op='intersects')
#     tropical_developed = points_in_polygons[points_in_polygons['2018']>12375]
#     tropical_developing = points_in_polygons[points_in_polygons['2018']<12375]
#
#     Developed_tac_trend_mean.append(tropical_developed.iloc[:,3:8].mean().values*0.5)
#     Developed_tac_trend_sd.append(tropical_developed.iloc[:, 3:8].std().values*0.5)
#
#     Developing_tac_trend_mean.append(tropical_developing.iloc[:,3:8].mean().values*0.5)
#     Developing_tac_trend_sd.append(tropical_developing.iloc[:, 3:8].mean().values*0.5)
#
# Developed_tac_trend_mean = np.array(Developed_tac_trend_mean)
# Developing_tac_trend_mean = np.array(Developing_tac_trend_mean)
# Developed_tac_trend_sd = np.array(Developed_tac_trend_sd)
# Developing_tac_trend_sd = np.array(Developing_tac_trend_sd)
#
# plt.figure(); plt.plot(cities['MAP'],cities['urban_core'],'o')
#
#
# fig, ax = plt.subplots(2,3, figsize=(9,6), sharey=True)
# x = np.array([0.9, 1.9,2.9,3.9])
# ax[0].errorbar(x, Tree_TRC_mean[:,0], yerr=Tree_TRC_sd[:,0], fmt="o", mfc='none',ms=8, color='b',lw=2.5)
# ax[0].errorbar(x+0.2, Tree_TRC_mean[:,1], yerr=Tree_TRC_sd[:,1], fmt="o", ms = 8, mfc='none', color='r',lw=2.5) # tropical # temporal
#
# ax[1].errorbar(x, grass_TRC_mean[:,0], yerr=grass_TRC_sd[:,0], fmt="o", mfc='none', ms=8,color='b',lw=2.5)
# ax[1].errorbar(x+0.2, grass_TRC_mean[:,1], yerr=grass_TRC_sd[:,1], fmt="o", ms=8, mfc='none', color='r',lw=2.5) # tropical # temporal