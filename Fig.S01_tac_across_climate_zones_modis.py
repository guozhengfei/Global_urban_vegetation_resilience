import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
import matplotlib;
# matplotlib.use('Qt5Agg')
import os
import numpy as np
import scipy.stats as st

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
relative_path = '/2_Output/tac_nadir_city_3zones.npz'
relative_path2 = '/2_Output/tac_landsat_city_3zones.npz'

TACs = np.load(current_dir+relative_path)['array1'] # v5,v4.2
ID = np.load(current_dir+relative_path)['array2'] # v5,v4.2
ID2 = np.load(current_dir+relative_path2)['array2']

TACs_mean = np.nanmean(TACs, axis=2)
df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_edge','rural_bgr']
df_tac['ID'] = ID
df_tac = df_tac[df_tac['ID'].isin(ID2)]

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')

# Reclassify Köppen climate zones using .loc
climate.loc[climate['koppen'] <= 3, 'koppen'] = 1
climate.loc[(climate['koppen'] > 3) & (climate['koppen'] <= 7), 'koppen'] = 2
climate.loc[(climate['koppen'] > 7) & (climate['koppen'] <= 16), 'koppen'] = 3
climate.loc[(climate['koppen'] > 16) & (climate['koppen'] <= 28), 'koppen'] = 4
climate.loc[(climate['koppen'] > 28) & (climate['koppen'] <= 30), 'koppen'] = 5

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
    tac_tropical_mean = tropical_city.iloc[:,3:6].mean().values
    tac_climate_mean.append(tac_tropical_mean)
    tac_tropical_sd = tropical_city.iloc[:, 3:6].std().values*0.5
    tac_climate_sd.append(tac_tropical_sd)

tac_climate_mean = np.array(tac_climate_mean)
tac_climate_sd = np.array(tac_climate_sd)

colors = ['#2166ac', '#67a9cf', '#b2182b']

fig, ax = plt.subplots(1,figsize=(4.5,2.5))
x = np.array([0.85, 1, 1.15])
ax.bar(x, tac_climate_mean[0,:], yerr=tac_climate_sd[0,:], width=0.15,color=colors) # tropical

ax.bar(x+1, tac_climate_mean[2,:], yerr=tac_climate_sd[1,:], width=0.15,color=colors) # temperate

ax.bar(x+2, tac_climate_mean[1,:], yerr=tac_climate_sd[2,:], width=0.15,color=colors) # boreal

ax.bar(x+3, tac_climate_mean[3,:], yerr=tac_climate_sd[3,:], width=0.15,color=colors) # arid

ax.bar(x, tac_climate_mean[0,:], yerr=tac_climate_sd[0,:], width=0.15,color=colors) # tropical
ax.set_ylim([0.10,0.30])

ax.set_xticks([1,2,3,4], ['Tropical', 'Boreal', 'Temperate', 'Arid'])
ax.tick_params(labelsize=12)

figToPath = current_dir + '/4_Figures/FigS01b_tac_by_koppen_modis'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)