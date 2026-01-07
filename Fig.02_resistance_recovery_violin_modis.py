import numpy as np
import matplotlib;
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
plt.rc('lines', linewidth=1.0)  # Increase default line width
plt.rc('axes', linewidth=1.0)   # Increase axes line width
plt.rc('grid', linewidth=1.0)   # Increase grid line width
plt.tick_params(width=1.0, labelsize=14)
plt.rc('xtick.major', size=4, width=1.0)    # Increase length and width of major ticks
plt.rc('ytick.major', size=4, width=1.0)    # Increase length and width of major ticks
plt.close()
from scipy.stats import gaussian_kde
import pandas as pd
import os


def remove_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[df_clean[col].between(lower_bound, upper_bound)]
    return df_clean

current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
urban_folder = current_dir + '/2_Output/Modis_recovery_resistance/'
folder = current_dir + '/2_Output/VI_Landsat/'
filenames = os.listdir(urban_folder)

# load Landsat TAC
relative_path = '/2_Output/tac_nadir_city_3zones_modis_no_disturbance.npz'
TACs = np.load(current_dir+relative_path)['array1'] # v5,v4.2
TACs[TACs<0]=np.nan
TACs_mean = TACs
ID = np.load(current_dir+relative_path)['array2'] # v5,v4.2
df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_edge','rural_bgr']
df_tac['ID'] = ID

IDs = []
for name in filenames:
    if name.startswith('dVI') and name.endswith('1sd.npy'):
        id = float(name.split('_')[1])
        IDs.append(id)

IDs = np.sort(IDs)

recovery_stats = []  # List to store recovery statistics
IDs_num = []
sd_names = ['2']
for sd_name in sd_names:
    for id in IDs[2:]:
        # if id in[539.0,606.0]: continue

        dt_vi = np.load(current_dir + '/2_Output/Modis_recovery_resistance/dVI_' + str(id) + '_smith_' + sd_name + 'sd.npy')

        urban_lab = np.load(urban_folder + 'urban_label_' + str(id) + '_.npy')

        dt_vi_inner = abs(np.nanmean(dt_vi[urban_lab == 2], axis=0))
        dt_vi_sub = abs(np.nanmean(dt_vi[urban_lab == 1], axis=0))
        dt_vi_rural = abs(np.nanmean(dt_vi[urban_lab == 0], axis=0))

        recovery_stats.append([id,dt_vi_inner,dt_vi_sub,dt_vi_rural])
    recovery_stats_arr = np.array(recovery_stats)

    np.nanmean(recovery_stats_arr,axis=0)
    recovery_df = pd.DataFrame(recovery_stats_arr)
    recovery_df.columns = ['ID','dt_vi_inner','dt_vi_sub','dt_vi_rural']
    columns_to_check = ['dt_vi_inner','dt_vi_sub','dt_vi_rural']
    recovery_df = remove_outliers(recovery_df, columns_to_check)
    recovery_df = recovery_df.dropna()



df_tac = pd.merge(recovery_df,df_tac,on='ID')
# Create figure with two subplots
fig, (ax2, ax1, ax3) = plt.subplots(1, 3, figsize=(11 * 0.72, 4 * 0.65))

df_tac['urban_core'][df_tac['urban_core']>np.nanpercentile(df_tac['urban_core'],98)]=np.nan
df_tac['rural_bgr'][df_tac['rural_bgr']>np.nanpercentile(df_tac['rural_bgr'],98)]=np.nan

# Data for plots
resistance_data = [df_tac['dt_vi_inner'], df_tac['dt_vi_rural']]
taced_data = [df_tac['urban_core'].dropna(), df_tac['rural_bgr'].dropna()]
colors = ['#01665e', '#8c510a']
labels = ['UC', 'RA']

# Plot resistance
parts1 = ax1.violinplot(resistance_data, showmeans=False, showmedians=False, showextrema=False)
for pc in parts1['bodies']:
    pc.set_facecolor(colors[0])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

ax1.boxplot(resistance_data, labels=labels, showfliers=False, patch_artist=True,
            boxprops=dict(facecolor='none', edgecolor='black'),
            medianprops=dict(color='black'))
ax1.set_ylabel('Resistance (kNDVI loss)')
ax1.set_ylim(resistance_data[1].min()-0.002, resistance_data[-1].max() * 1.15)
# Add labels on top of each violin bar
ax1.text(1.5, ax1.get_ylim()[1] * 0.95, 'p>0.05', ha='center', va='top', fontsize=8, fontweight='bold')

# Plot recovery
parts2 = ax2.violinplot(taced_data, showmeans=False, showmedians=False, showextrema=False)
for pc in parts2['bodies']:
    pc.set_facecolor(colors[1])
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

ax2.boxplot(taced_data, labels=labels, showfliers=False, patch_artist=True,
            boxprops=dict(facecolor='none', edgecolor='black'),
            medianprops=dict(color='black'))
ax2.set_ylabel('Resilience (TAC$_{ED}$)')
ax2.set_ylim(taced_data[0].min()-0.01, taced_data[-1].max()*1.15)
# Add labels on top of each violin bar
ax2.text(1.5, ax2.get_ylim()[1] * 0.95, 'p<0.001', ha='center', va='top', fontsize=8, fontweight='bold')
# ax2.text(2, ax2.get_ylim()[1] * 0.95, 'b', ha='center', va='top', fontsize=10, fontweight='bold')

df_merge = df_tac
tac_uc = df_merge['urban_core']
tac_rb = df_merge['rural_bgr']

dt_vi_uc = df_merge['dt_vi_inner']
dt_vi_rb = df_merge['dt_vi_rural']

mask = np.isnan(tac_uc-tac_rb-dt_vi_uc-dt_vi_rb)
# Calculate normalized differences
norm_diff_tac = (tac_uc - tac_rb) / abs(tac_uc - tac_rb).max()
norm_diff_dt_vi = (dt_vi_uc - dt_vi_rb) / abs(dt_vi_uc - dt_vi_rb).max()

x = np.linspace(-1, 1, 200)
kde1 = gaussian_kde(norm_diff_tac[~mask])
y1 = kde1(x)

kde2 = gaussian_kde(norm_diff_dt_vi)
y2 = kde2(x)

ax3.plot(x, y1, color=colors[1], linewidth=2)
ax3.plot(x, y2, color=colors[0], linewidth=2)
ax3.fill_between(x[:100], 0, np.minimum(y1[:100], y2[:100]), color='r', alpha=0.5, label='Both < 0'); #np.minimum(y1, y2),

ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7)
ax3.set_ylabel('Density')
ax3.set_xlabel('Normalized urban-rural differences')

sum((norm_diff_tac<0.1) & (norm_diff_dt_vi<0.1))
df_tac['tradeoff'] = ((norm_diff_tac<-0.1) & (norm_diff_dt_vi<-0.1)).astype(int)
# 237/665

# x_fit10 = np.array([dt_vi_uc.min(), dt_vi_uc.max()])
x_fit10 = x_fit12 = np.array([dt_vi_rb.min(), dt_vi_rb.max()])

plt.tight_layout()
plt.savefig(os.path.join(current_dir, '4_Figures', 'recovery_resistance_TAC_modis.png'), dpi=900)
plt.show()

import scipy.stats as st
# Perform statistical test
print("T-test results (inner vs rural recovery):")
print(st.ttest_rel(dt_vi_uc, dt_vi_rb))
print(st.ttest_rel(tac_uc[~mask], tac_rb[~mask]))

import geopandas as gpd
import geodatasets
climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate['MAP'] = climate['MAP']*24*1000

coastline = gpd.read_file(geodatasets.get_path('naturalearth.land'))
tropical = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_tropical.shp')
arid = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_arid.shp')
temperate = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_temperate.shp')
boreal = gpd.read_file(current_dir + '/1_Input/shps/Shp/koppen_boreal.shp')
cities = gpd.read_file(current_dir + '/1_Input/shps/Shp/points_citis.shp')

cities = pd.merge(cities, df_tac, on='ID')
cities = pd.merge(cities, climate, on='ID')
cities_1 = cities.loc[cities['tradeoff']==0]
cities_2 = cities.loc[cities['tradeoff']==1]
fig, ax = plt.subplots(1, 2, figsize=(11 * 0.85, 4 * 0.65), gridspec_kw={'width_ratios': [2.35, 0.75]})
# Plot world coastline using land boundaries
coastline.boundary.plot(ax=ax[0], color='k', linewidth=0.5, zorder=10)

# Plot climate zones and cities with core_rural_diff
tropical.plot(ax=ax[0], color='#F78A5D', edgecolor='none', alpha=0.4)
temperate.plot(ax=ax[0], color='#AAD664', edgecolor='none', alpha=0.4)
arid.plot(ax=ax[0], color='#FFC96E', edgecolor='none', alpha=0.5)
boreal.plot(ax=ax[0], color='lightgrey', edgecolor='none', alpha=0.4)

cities_1.plot('tradeoff', ax=ax[0], marker='o', markersize=7, cmap='Blues',vmin=-0.5,vmax=0.5)
cities_2.plot('tradeoff', ax=ax[0], marker='o', markersize=10, cmap='Reds',vmin=-0,vmax=1.5)

ax[0].set_ylim([-54, 80])
ax[0].set_xlim([-149, 170])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel('')
ax[0].set_ylabel('')

irri_df = pd.read_csv(os.path.join('..', '2_Output', 'pure_veg', 'monthly_landsat_et_PM_global.csv'))
irri_df['irri_mean'] = np.nanmean(irri_df.iloc[:,-12:],axis=1)
irri_uc = irri_df[irri_df['type']=='core'][['id','irri_mean']]
irri_uc2 = irri_uc.groupby('id',as_index=False).aggregate('mean')

irri_bg = irri_df[irri_df['type']=='bg'][['id','irri_mean']]
irri_bg2 = irri_bg.groupby('id',as_index=False).aggregate('mean')

irri_bg2['diff'] = (irri_uc2['irri_mean']-irri_bg2['irri_mean'])*10 # mm/year
irri_bg2.columns = ['ID','irri_mean','irri_diff']
df_tac = pd.merge(df_tac,irri_bg2,on='ID')
df_tac2 = df_tac.loc[df_tac['tradeoff']==1]

ax[1].plot(df_tac['irri_diff'],df_tac['urban_core']-df_tac['rural_bgr'],'o',c=colors[1],mfc='None',ms=5,alpha=0.5)
ax2 = ax[1].twinx()
ax2.plot(df_tac['irri_diff'],df_tac['dt_vi_inner']-df_tac['dt_vi_rural'],'o',c=colors[0],mfc='None',ms=5,alpha=0.5)

mask2 = np.isnan(df_tac['urban_core']-df_tac['rural_bgr']-df_tac['irri_diff']).values
x_fit10 = np.array([df_tac['irri_diff'].min(), df_tac['irri_diff'].max()])

slope20, intercept20, r_value20, p_value20, std_err20 = st.linregress(df_tac['irri_diff'].values[~mask2],(df_tac['urban_core']-df_tac['rural_bgr']).values[~mask2])
slope22, intercept22, r_value22, p_value22, std_err22 = st.linregress(df_tac['irri_diff'].values[~mask2],(df_tac['dt_vi_inner']-df_tac['dt_vi_rural']).values[~mask2])


ax[1].plot(x_fit10, slope20 * x_fit10 + intercept20, colors[1], linewidth=1.5)
ax2.plot(x_fit10, slope22 * x_fit10 + intercept22, colors[0], linewidth=1.5)
ax[1].tick_params(axis='y', colors=colors[1])
ax2.tick_params(axis='y', colors=colors[0])

ax[1].set_xlabel('Urban irrigation (mm/year)')
ax[1].set_ylabel('ΔResilience',c=colors[1])
ax2.set_ylabel('ΔResistance',c=colors[0])
ax[1].set_xlim([-500,500])
fig.tight_layout(h_pad=0.05)
plt.savefig(os.path.join(current_dir, '4_Figures', 'recovery_resistance_TAC_modis_d_e.png'), dpi=900)
