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
TACs = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array1'] # v5,v4.2
ID = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array2'] # v5,v4.2

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

df_tac = pd.DataFrame(urban_to_rural_slopes)

df_tac.columns=['urban_core','urban_edge','rural_bgr']
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
    tac_tropical_mean = tropical_city.iloc[:,3:6].mean().values
    tac_climate_mean.append(tac_tropical_mean)
    tac_tropical_sd = tropical_city.iloc[:, 3:6].std().values*0.5
    tac_climate_sd.append(tac_tropical_sd)

tac_climate_mean = np.array(tac_climate_mean)
tac_climate_sd = np.array(tac_climate_sd)

colors = ['#7fc97f', '#beaed4', '#fdc086']

fig, ax = plt.subplots(1,figsize=(9,3.3))
x = np.array([0.85, 1, 1.15])
ax.bar(x, tac_climate_mean[0,:], yerr=tac_climate_sd[0,:], width=0.15,color=colors) # tropical

ax.bar(x+1, tac_climate_mean[1,:], yerr=tac_climate_sd[1,:], width=0.15,color=colors) # temperate

ax.bar(x+2, tac_climate_mean[2,:], yerr=tac_climate_sd[2,:], width=0.15,color=colors) # boreal

ax.bar(x+3, tac_climate_mean[3,:], yerr=tac_climate_sd[3,:], width=0.15,color=colors) # arid

ax.bar(x, tac_climate_mean[0,:], yerr=tac_climate_sd[0,:], width=0.15,color=colors) # tropical

ax.set_xticks([1,2,3,4], ['Tropical', 'Temperate', 'Boreal', 'Arid'])
ax.tick_params(labelsize=12)

figToPath = current_dir + '/4_Figures/Fig02b_temporal_tac_by_koppen'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
# plt.close(fig)

fig, ax = plt.subplots(2,3,figsize=(9*0.9,6*0.9))
ax[0,0].plot(cities['MAT'],cities['urban_core'],'o',mfc='none')
mask = np.isnan(cities['MAT']) | np.isnan(cities['urban_core'])
st.linregress(cities['MAT'][~mask],cities['urban_core'][~mask])
ax[1,0].plot(cities['MAP'],cities['urban_core'],'o',mfc='none')
st.linregress(cities['MAP'][~mask],cities['urban_core'][~mask])

ax[0,1].plot(cities['MAT'],cities['urban_edge'],'o',mfc='none')
ax[1,1].plot(cities['MAP'],cities['urban_edge'],'o',mfc='none')

ax[0,2].plot(cities['MAT'],cities['rural_bgr'],'o',mfc='none')
ax[1,2].plot(cities['MAP'],cities['rural_bgr'],'o',mfc='none')


fig, ax = plt.subplots(2,1,figsize=(3*0.9,6*0.9))
ax[0].plot(cities['urban_core'],cities['urban_edge'],'o',mfc='none')
ax[0].plot([-0.2,0.2],[-0.2,0.2],'r')
ax[1].plot(cities['urban_core'],cities['rural_bgr'],'o',mfc='none')
ax[1].plot([-0.2,0.2],[-0.2,0.2],'r')
mask = np.isnan(cities['MAT']) | np.isnan(cities['urban_core'])
st.linregress(cities['MAT'][~mask],cities['urban_core'][~mask])

fig, ax = plt.subplots(2,1,figsize=(3*0.9,6*0.9))
ax[0].plot(cities['MAP'],cities['urban_core']-cities['rural_bgr'],'o',mfc='none')
mask = np.isnan(cities['MAP']) | np.isnan(cities['urban_core'])
st.linregress(cities['MAP'][~mask],(cities['urban_core']-cities['rural_bgr'])[~mask])

ax[1].plot(cities['MAT'],cities['urban_core']-cities['rural_bgr'],'o',mfc='none')
mask = np.isnan(cities['MAT']) | np.isnan(cities['urban_core'])
st.linregress(cities['MAT'][~mask],(cities['urban_core']-cities['rural_bgr'])[~mask])


fig.tight_layout()