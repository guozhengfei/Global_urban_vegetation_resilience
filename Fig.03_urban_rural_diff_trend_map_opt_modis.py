import geopandas as gpd
import geodatasets
import pandas as pd
import matplotlib; matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import os
import numpy as np
import seaborn as sns
from scipy import stats

plt.rc('font', family='Arial')
plt.rc('lines', linewidth=1.0)  # Increase default line width
plt.rc('axes', linewidth=1.0)   # Increase axes line width
plt.rc('grid', linewidth=1.0)   # Increase grid line width
plt.tick_params(width=1.05, labelsize=14)
plt.rc('xtick.major', size=4, width=1.0)    # Increase length and width of major ticks
plt.rc('ytick.major', size=4, width=1.0)    # Increase length and width of major ticks
plt.close()

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
TACs = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array1'] #
ID2 = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array2'] #
ID = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array2'] #

TACs_mean = np.nanmean(TACs, axis=2)
tac_global_mean = np.nanmean(TACs_mean, axis=0)

df_tac = pd.DataFrame(TACs_mean)
df_tac.columns = ['urban_core','urban_edge','rural_bgr']

TACs_diff = (TACs[:,-1,:]-TACs[:,0,:])/10

# plt.figure(); plt.plot(TACs[-1,0,:])
# Calculate slopes for each city's time series
slopes = np.zeros(TACs_diff.shape[0])  # Array to store slopes
p_values = np.zeros(TACs_diff.shape[0])  # Array to store p-values
time = np.arange(TACs_diff.shape[1])  # Time array

for i in range(TACs_diff.shape[0]):
    # Get current time series
    y = TACs_diff[i, :]

    # Remove NaN values
    mask = ~np.isnan(y)
    y_clean = y[mask]
    t_clean = time[mask]

    # Calculate slope if we have enough valid data points (e.g., > 10)
    if len(y_clean) > 10:
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(t_clean, y_clean)
        slopes[i] = slope
        p_values[i] = p_value
    else:
        slopes[i] = np.nan
        p_values[i] = np.nan

# Add slopes to df_tac
df_tac['tac_diff_slope'] = slopes
df_tac['tac_diff_pvalue'] = p_values
# plt.figure(); plt.hist(df_tac['tac_diff_slope_sig'],50);
# Filter significant trends (p < 0.05)
df_tac['tac_diff_slope_sig'] = df_tac['tac_diff_slope']*-1 # urban_core - rural_bgr
df_tac.loc[df_tac['tac_diff_pvalue'] >= 0.05, 'tac_diff_slope_sig'] = 0
df_tac['ID'] = ID
df_tac = df_tac[df_tac['ID'].isin(ID2)]
df_tac.to_csv(current_dir + '/2_Output/tac_diff_trend.csv', index=False)

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate['MAT'] = climate['MAT']-273.15
climate['MAP'] = climate['MAP']*24*1000

coastline = gpd.read_file(geodatasets.get_path('naturalearth.land'))
tropical = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_tropical.shp')
arid = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_arid.shp')
temperate = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_temperate.shp')
boreal = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_boreal.shp')
cities = gpd.read_file(current_dir + '/1_Input/shps/Shp/points_citis.shp')

cities = pd.merge(cities, df_tac, on='ID')
cities = pd.merge(cities, climate, on='ID')

fig, ax = plt.subplots(1, 3, figsize=(10.5*0.8, 2.9*0.8), gridspec_kw={'width_ratios': [2, 0.5, 1]})
coastline.boundary.plot(ax=ax[0], color='k', linewidth=0.5, zorder=10)

# Plot climate zones and cities with core_rural_diff
tropical.plot(ax=ax[0], color='#F78A5D', edgecolor='none', alpha=0.4)
temperate.plot(ax=ax[0], color='#AAD664', edgecolor='none', alpha=0.4)
arid.plot(ax=ax[0], color='#FFC96E', edgecolor='none', alpha=0.4)
boreal.plot(ax=ax[0], color='lightgrey', edgecolor='none', alpha=0.4)

cities.plot('tac_diff_slope_sig', ax=ax[0], marker='o', markersize=10, cmap='RdBu', vmin=-0.0005, vmax=0.0005)

ax[0].set_ylim([-54, 80])
ax[0].set_xlim([-149, 170])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel('')
ax[0].set_ylabel('')
# Calculate mean core_rural_diff for each 5-degree latitude bin
cities['latitude_bin'] = (cities.geometry.y // 5) * 5
latitude_bins = cities.groupby('latitude_bin')['tac_diff_slope_sig'].agg(['mean', 'std']).reset_index()

# Remove the bin values at -40 latitude
latitude_bins = latitude_bins[latitude_bins['latitude_bin'] != -40]

# Rotate the second panel 90 degrees and plot
ax[1].plot(latitude_bins['mean'], latitude_bins['latitude_bin'], color='k',lw=1.5)
ax[1].fill_betweenx(latitude_bins['latitude_bin'], latitude_bins['mean'] - 0.5 * latitude_bins['std'], latitude_bins['mean'] + 0.5 * latitude_bins['std'], color='k', alpha=0.2)
ax[1].invert_yaxis()  # Invert y-axis to match the map
ax[1].set_ylim([-50, 70])
ax[1].set_xlim([-0.0005, 0.0005])

# Modify ax[1] x-axis ticks
# ax[1].set_xticks([-0.0004, -0.0002, 0, 0.0002, 0.0004])
# ax[1].set_xticklabels(['-4', '-2', '0', '2', '4'])

# Create a heatmap for core_rural_diff distribution in MAT and MAP space with 9x9 grid
cities['MAT_bin'] = pd.cut(cities['MAT'], bins=9)
cities['MAP_bin'] = pd.cut(cities['MAP'], bins=9)
heatmap_data = cities.pivot_table(index='MAT_bin', columns='MAP_bin', values='tac_diff_slope_sig', aggfunc='mean')

# Prepare data for pcolor
heatmap_data = heatmap_data.values
x_edges = np.linspace(cities['MAP'].min(), cities['MAP'].max(), heatmap_data.shape[1] + 1)
y_edges = np.linspace(cities['MAT'].min(), cities['MAT'].max(), heatmap_data.shape[0] + 1)

# Plot using pcolor
c = ax[2].pcolor(x_edges, y_edges, heatmap_data, cmap='RdBu', edgecolors='k', linewidths=1, vmin=-0.0005, vmax=0.0005)
cbar = fig.colorbar(c, ax=ax[2], label='Core-Rural Difference trend')
cbar.ax.set_yticklabels([f'{x:.1f}' for x in cbar.get_ticks()*10000])

ax[2].set_xlabel('MAP (mm/year)')
ax[2].set_ylabel('MAT (°C)')
ax[2].tick_params(left=False, bottom=False)  # Hide the ticks and labels

figToPath = current_dir + '/4_Figures/Fig03_urban_rural_diff_trend_modis'
fig.tight_layout()
fig.savefig(figToPath, dpi=900)
# plt.close(fig)
