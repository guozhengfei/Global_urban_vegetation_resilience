import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
# import matplotlib; matplotlib.use('Qt5Agg')
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

irri_df = pd.read_csv(os.path.join('..', '2_Output', 'pure_veg', 'monthly_landsat_et_PM_global.csv'))
irri_df['irri_mean'] = np.nanmean(irri_df.iloc[:,-12:],axis=1)
irri_uc = irri_df[irri_df['type']=='core'][['id','irri_mean']]
irri_uc2 = irri_uc.groupby('id',as_index=False).aggregate('mean')

irri_bg = irri_df[irri_df['type']=='bg'][['id','irri_mean']]
irri_bg2 = irri_bg.groupby('id',as_index=False).aggregate('mean')

irri_bg2['diff'] = (irri_uc2['irri_mean']-irri_bg2['irri_mean'])*10 # mm/year
irri_bg2.columns = ['ID','irri_mean','irri_diff']
df_tac = pd.merge(df_tac,irri_bg2,on='ID')
# urban_factors = pd.read_csv(os.path.join('..', '2_Output', 'drivers', 'urban_factors_effect.csv'))
# df_tac = pd.merge(df_tac,urban_factors,on='ID')
#
# df_tac['LE_core'] = df_tac['Rn_core'] + df_tac['Q_core'] - df_tac['H_core'] - df_tac['Rn_core'] * 0.25
# df_tac['LE_bg'] = df_tac['Rn_bg'] + df_tac['Q_bg'] - df_tac['H_bg'] - df_tac['Rn_bg'] * 0.25
# df_tac['irri']=df_tac['LE_core']-df_tac['LE_bg']

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate['MAT'] = climate['MAT']-273.15
climate['MAP'] = climate['MAP']*24*1000

import geodatasets
# world = gpd.read_file(gpd.geodatasets.get_path('naturalearth_lowres'))
tropical = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_tropical.shp')
arid = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_arid.shp')
temperate = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_temperate.shp')
boreal = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_boreal.shp')
cities = gpd.read_file(current_dir + '/1_Input/shps/Shp/points_citis.shp')

cities = pd.merge(cities, df_tac, on='ID')
cities = pd.merge(cities, climate, on='ID')

fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9), gridspec_kw={'width_ratios': [2, 0.5, 1]})

# Plot climate zones and cities with core_rural_diff
tropical.plot(ax=ax[0], color='#F78A5D', edgecolor='none', alpha=0.4)
temperate.plot(ax=ax[0], color='#AAD664', edgecolor='none', alpha=0.4)
arid.plot(ax=ax[0], color='#FFC96E', edgecolor='none', alpha=0.4)
boreal.plot(ax=ax[0], color='lightgrey', edgecolor='none', alpha=0.4)

cities['core_rural_diff'] = cities['irri_diff']
cities.plot('core_rural_diff', ax=ax[0], marker='o', markersize=10, cmap='RdBu', vmin=-400, vmax=400)

ax[0].set_ylim([-54, 80])
ax[0].set_xlim([-149, 170])

# Calculate mean core_rural_diff for each 5-degree latitude bin
cities['latitude_bin'] = (cities.geometry.y // 5) * 5
latitude_bins = cities.groupby('latitude_bin')['core_rural_diff'].agg(['mean', 'std']).reset_index()

# Remove the bin values at -40 latitude
latitude_bins = latitude_bins[latitude_bins['latitude_bin'] != -40]

# Rotate the second panel 90 degrees and plot
ax[1].plot(latitude_bins['mean'], latitude_bins['latitude_bin'], color='k',lw=1.5)
ax[1].fill_betweenx(latitude_bins['latitude_bin'], latitude_bins['mean'] - 0.5 * latitude_bins['std'], latitude_bins['mean'] + 0.5 * latitude_bins['std'], color='k', alpha=0.2)
ax[1].invert_yaxis()  # Invert y-axis to match the map
ax[1].set_ylim([-50, 70])

# Create a heatmap for core_rural_diff distribution in MAT and MAP space with 9x9 grid
cities['MAT_bin'] = pd.cut(cities['MAT'], bins=9)
cities['MAP_bin'] = pd.cut(cities['MAP'], bins=9)
heatmap_data = cities.pivot_table(index='MAT_bin', columns='MAP_bin', values='core_rural_diff', aggfunc='mean')

# Prepare data for pcolor
heatmap_data = heatmap_data.values
x_edges = np.linspace(cities['MAP'].min(), cities['MAP'].max(), heatmap_data.shape[1] + 1)
y_edges = np.linspace(cities['MAT'].min(), cities['MAT'].max(), heatmap_data.shape[0] + 1)

# Plot using pcolor
c = ax[2].pcolor(x_edges, y_edges, heatmap_data, cmap='RdBu', edgecolors='k', linewidths=0.5, vmin=-400, vmax=400)
fig.colorbar(c, ax=ax[2])

ax[2].set_xlabel('MAP (mm/year)')
ax[2].set_ylabel('MAT (°C)')
ax[2].tick_params(left=False, bottom=False)  # Hide the ticks and labels

figToPath = current_dir + '/4_Figures/FigS03_irri_mat_map_landsat_v2'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
# plt.close(fig)

from scipy import stats
import seaborn as sns
AI = pd.read_csv(current_dir + '/2_Output/AI_csv.csv')
df_tac = pd.merge(df_tac,AI,on='ID')
df_clean = df_tac.dropna(subset=['AI', 'irri_diff'])

fig, ax = plt.subplots(1,figsize=(4.5*0.9, 3*0.9))

scatter = ax.scatter(df_clean['AI'], df_clean['irri_diff'],
                    alpha=0.5, s=20, label='Data points')

# Calculate and plot linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['AI'], 
                                                              df_clean['irri_diff'])
x_range = np.linspace(df_clean['AI'].min(), df_clean['AI'].max(), 100)
ax.plot(x_range, slope * x_range + intercept, 'r-', 
        label=f'Linear fit (R² = {r_value**2:.2f})')

# Customize plot
ax.set_xlabel('Aridity Index (AI)', fontsize=12)
ax.set_ylabel('Irrigation intensity', fontsize=12)
ax.set_ylim([-500,800])


# Add statistics text box
stats_text = (f'n = {len(df_clean)}\n'
             f'y = {slope:.3f}x + {intercept:.3f}\n'
             f'p-value = {p_value:.2e}\n'
             f'{r_value}')
print(stats_text)

# Adjust layout and save
figToPath = current_dir + '/4_Figures/FigS03b_irri_AI_v2'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
