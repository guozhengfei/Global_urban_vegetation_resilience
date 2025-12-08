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

Developed_tac_trend_mean = []
Developing_tac_trend_mean = []
Developed_tac_trend_sd = []
Developing_tac_trend_sd = []

points_in_polygons = gpd.sjoin(cities, world_countries, how="left", op='intersects')
tropical_developed = points_in_polygons[points_in_polygons['2018']>12375]
tropical_developing = points_in_polygons[points_in_polygons['2018']<12375]

Developed_tac_trend_mean = tropical_developed.iloc[:,3:6].median().values
Developed_tac_trend_sd = tropical_developed.iloc[:, 3:6].std().values*0.5

Developing_tac_trend_mean = tropical_developing.iloc[:,3:6].median().values
Developing_tac_trend_sd = tropical_developing.iloc[:, 3:6].std().values*0.5

colors = ['#7fc97f', '#beaed4', '#fdc086',  '#80b1d3', '#fb8072']

fig, ax = plt.subplots(1,figsize=(9,3.3))
x = np.array([0.5, 1, 1.5])
ax.bar(x, Developed_tac_trend_mean, yerr=Developed_tac_trend_sd, width=0.15,color=colors,ec='k') # tropical

ax.bar(x+0.15, Developing_tac_trend_mean, yerr=Developing_tac_trend_sd, width=0.15,color=colors,hatch = '//',ec='k') # temperate


