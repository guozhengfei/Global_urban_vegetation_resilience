import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;
matplotlib.use('Qt5Agg')
import scipy.stats as st
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
import geopandas as gpd
from scipy.ndimage import convolve1d
import os
import rasterio
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def ar1_series_5yr(array):
    yrs = 5  # 3,5,7
    import pandas as pd
    import numpy.ma as ma
    def calc_ar1(x):
        return ma.corrcoef(ma.masked_invalid(x[:-1]), ma.masked_invalid(x[1:]))[0, 1]

    bands_year = 12
    t = bands_year * yrs

    ar1 = pd.Series(array).rolling(t, min_periods=6 * yrs, center=True).apply(calc_ar1).values
    return ar1

## main ##
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
coord = pd.read_csv(current_dir+'/2_Output/PICS_sites.csv')[['Lon','Lat']].values
VI_array0 = np.load(current_dir + '/2_Output/VI_values_PICS_sites.npy',allow_pickle=True).astype(float)
VI_array0_mean = np.nanmean(VI_array0,axis=1)
mask = (VI_array0_mean<0.11)&(VI_array0_mean>0.07)

VI_array = VI_array0[mask,:]
VI_array[:,12*12:12*12+12] = (VI_array[:,11*12:11*12+12]+VI_array[:,13*12:13*12+12])/2
V_median = np.nanmedian(VI_array,axis=0)
V_median[V_median>0.105] = np.nan
V_median[V_median<0.057] = np.nan

# Create a single figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10*0.8, 8.5*0.8), sharex=True)
time_axis = np.linspace(2001, 2023+11/12, 276)

# Plot 1: Original data
ax1.plot(time_axis, VI_array.T, 'o', mfc='none', ms=3, alpha=0.5)
ax1.plot(time_axis, V_median, 'ko-', alpha=0.8, linewidth=2)
ax1.set_ylim(0.04, 0.13)
ax1.set_ylabel('VI (Original)')
ax1.set_title('Landsat Sensor Bias Analysis at PICS Sites')
ax1.grid(True, alpha=0.3)

# bands_year = 12
# V_median_rsp = np.reshape(V_median, ( int(V_median.shape[0] / bands_year), bands_year))
# plt.figure(); plt.plot(V_median_rsp)
# plt.figure(); plt.plot(np.nanstd(V_median_rsp,axis=1))

ser = V_median*1
EVI_yr = np.zeros_like(ser)

yr_num = 23
bands_year = 12
for year in range(yr_num):
    st = year * bands_year
    ed = st + bands_year
    evi_year = np.nanmean(ser[st:ed])
    evi_year_rep = [evi_year]*int(bands_year)
    EVI_yr[st:ed] = evi_year_rep
rm_offline = ser - EVI_yr

L8 = V_median[14*12+3:]
correct_VI_median = rm_offline + np.nanmean(L8)
offsets = correct_VI_median - V_median
# plt.figure(); plt.plot(correct_VI_median); plt.plot(V_median)

# Plot 2: Bias-corrected data
VI_array_correct = VI_array+np.repeat(offsets[:,np.newaxis],VI_array.shape[0],axis=1).T
ax2.plot(time_axis, VI_array_correct.T, 'o', mfc='none', ms=3, alpha=0.5)
ax2.plot(time_axis, np.nanmedian(VI_array_correct,axis=0), 'ko-', alpha=0.8, linewidth=2)
ax2.set_ylim(0.04, 0.13)
ax2.set_ylabel('VI (Bias-corrected)')
ax2.grid(True, alpha=0.3)

# remove long-term mean
ser = VI_array_correct * 1
EVI_yr = np.zeros_like(ser)
for year in range(yr_num):
    st = year * bands_year
    ed = st + bands_year
    evi_year = np.nanmean(ser[:, st:ed], axis=1)
    evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
    EVI_yr[:, st:ed] = evi_year_rep
rm_offline = ser - EVI_yr

# remove seasonality
Evi_sea_rep = np.zeros_like(rm_offline)
for yr in range(yr_num):
    start_index = (yr - 3) * bands_year
    if start_index < 0: start_index = 0
    end_index = (yr + 4) * bands_year
    if end_index > bands_year * yr_num: end_index = bands_year * yr_num
    data_i = rm_offline[:, start_index:end_index]
    Evi_sea = np.nanmean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)),
                      axis=1)
    Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea

res = rm_offline - Evi_sea_rep
# plt.figure(); plt.plot(np.nanmean(res,axis=0))

# Plot 3: Residuals after detrending and deseasoning
ax3.plot(time_axis, res.T, 'o', mfc='none', ms=3, alpha=0.5)
ax3.plot(time_axis, np.nanmedian(res,axis=0), 'ko-', alpha=0.8, linewidth=2)
ax3.set_ylim(-0.05, 0.05)
ax3.set_ylabel('VI Residuals')
ax3.set_xlabel('Year')
ax3.grid(True, alpha=0.3)

# Adjust layout and show
plt.tight_layout()
plt.savefig(os.path.join('..', '4_Figures', 'PICS_sites_VI_bias_elimition.png'),
            dpi=600, bbox_inches='tight')
plt.show()


def create_global_map(points_df):
    fig = plt.figure(figsize=(10*0.75, 3.2*0.8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    # Set global range
    ax.set_global()

    # Plot site locations
    colors = plt.cm.Set1(np.linspace(0, 1, len(points_df)))

    for idx, row in points_df.iterrows():
        ax.scatter(row['Lon'], row['Lat'],
                   color=colors[idx], s=50, alpha=0.8,
                   transform=ccrs.PlateCarree(),
                   edgecolors='black', linewidth=0.5)

        # # Add site labels with offset
        # ax.text(row['lon'] + 2, row['lat'], row['site'],
        #         transform=ccrs.PlateCarree(),
        #         fontsize=8, ha='left', va='center')

    # Add title and legend
    ax.set_ylim([-10,70])

    plt.tight_layout()
    return fig

# 创建全球站点分布图
points_df = pd.read_csv(current_dir + '/2_Output/PICS_sites.csv')
fig2 = create_global_map(points_df)
plt.savefig(os.path.join('..', '4_Figures', 'PICS_sites_map.png'),
            dpi=600, bbox_inches='tight')
plt.show()