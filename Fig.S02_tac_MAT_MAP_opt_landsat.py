import geopandas as gpd
import pandas as pd
import matplotlib; matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import os
import numpy as np
import scipy.stats as st
plt.rc('font', family='Arial')
plt.rc('lines', linewidth=1.05)  # Increase default line width
plt.rc('axes', linewidth=1.05)   # Increase axes line width
plt.rc('grid', linewidth=1.05)   # Increase grid line width
plt.tick_params(width=1.05, labelsize=14)
plt.rc('xtick.major', size=4, width=1.05)    # Increase length and width of major ticks
plt.rc('ytick.major', size=4, width=1.05)    # Increase length and width of major ticks
plt.close()

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
TACs = np.load(current_dir+'/2_Output/tac_landsat_city_3zones.npz')['array1'] #
ID = np.load(current_dir+'/2_Output/tac_landsat_city_3zones.npz')['array2'] #
ID2 = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array2'] #

AI = pd.read_csv(current_dir + '/2_Output/AI_csv.csv')
TACs_mean = np.nanmean(TACs, axis=2)
tac_global_mean = np.nanmean(TACs_mean, axis=0)

df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_edge','rural_bgr']
df_tac['ID'] = ID
df_tac = df_tac[df_tac['ID'].isin(ID2)]

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate['MAT'] = climate['MAT']-273.15
climate['MAP'] = climate['MAP']*24*1000

cities = gpd.read_file(current_dir + '/1_Input/shps/Shp/points_citis.shp')
cities = pd.merge(cities, df_tac, on='ID')
cities = pd.merge(cities, climate, on='ID')
cities = pd.merge(cities, AI, on='ID')

fig, axes = plt.subplots(1, 3, figsize=(11.3*0.68, 3.8*0.68))  # 1x4 grid of subplots

scatter0 = axes[0].scatter(cities['MAT'], cities['urban_core'], c=cities['MAP'],
                          cmap='RdBu_r',
                          s=25,vmin=50, vmax=1500,
                          alpha=0.7)
# Calculate binned means for smooth line
x_values = cities['MAT'].values
y_values = cities['urban_core'].values
fig.colorbar(scatter0, ax=axes[0], orientation='horizontal', pad=0.2)

# For MAT subplot
mask_mat = ~np.isnan(cities['MAT']) & ~np.isnan(cities['urban_core'])
slope_mat, intercept_mat, r_mat, p_mat, _ = st.linregress(
    cities['MAT'][mask_mat], cities['urban_core'][mask_mat])
x_fit_mat = np.array([cities['MAT'].min(), cities['MAT'].max()])
axes[0].plot(x_fit_mat, slope_mat * x_fit_mat + intercept_mat, 'k-', lw=1.5)

scatter1 = axes[1].scatter(cities['MAP'], cities['urban_core'], c=cities['MAT'],
                          cmap='BrBG_r',
                          s=25,vmin=2, vmax=28,
                          alpha=0.7)
fig.colorbar(scatter1, ax=axes[1], orientation='horizontal', pad=0.2)

# For MAP subplot
mask_map = ~np.isnan(cities['MAP']) & ~np.isnan(cities['urban_core'])
slope_map, intercept_map, r_map, p_map, _ = st.linregress(
    cities['MAP'][mask_map], cities['urban_core'][mask_map])
x_fit_map = np.array([cities['MAP'].min(), cities['MAP'].max()])
axes[1].plot(x_fit_map, slope_map * x_fit_map + intercept_map, 'k-', lw=1.5)

x = cities['AI'].values
y = cities['urban_core'].values#-cities['rural_bgr'].values
scatter2 = axes[2].scatter(x, y, c=cities['AI'],
                          cmap='PuOr_r',
                          s=25,vmin=0, vmax=0.25,
                          alpha=0.7)
fig.colorbar(scatter2, ax=axes[2], orientation='horizontal', pad=0.2)

# For AI subplot
mask_ai = ~np.isnan(cities['AI']) & ~np.isnan(cities['urban_core'])
slope_ai, intercept_ai, r_ai, p_ai, _ = st.linregress(
    cities['AI'][mask_ai], cities['urban_core'][mask_ai])
x_fit_ai = np.array([cities['AI'].min(), cities['AI'].max()])
axes[2].plot(x_fit_ai, slope_ai * x_fit_ai + intercept_ai, 'k-', lw=1.5)

mask = np.isnan(x) | np.isnan(y)
st.linregress(x[~mask], y[~mask])

figToPath = current_dir + '/4_Figures/FigS02_tac_MAT_MAP_AI_landsat'
fig.tight_layout()
fig.savefig(figToPath, dpi=900, bbox_inches='tight')
# plt.close(fig)

fig, axes = plt.subplots(1, 1, figsize=(4.8*0.7, 3.8*0.7))  # 1x4 grid of

