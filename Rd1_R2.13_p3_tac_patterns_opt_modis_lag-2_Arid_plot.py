import geopandas as gpd
import pandas as pd
import matplotlib;
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
plt.close()
import os
import numpy as np
import scipy.stats as st

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
relative_path = '/2_Output/tac_nadir_city_3zones_all_lag2.npz'
original_relative_path = '/2_Output/tac_nadir_city_3zones_all.npz'
arid_shp = current_dir + '/1_Input/shps/Shp/koppen_arid.shp'
city_shp = current_dir + '/1_Input/shps/Shp/points_citis.shp'


def load_tac_3zone_df(path):
    data = np.load(path)
    tac = data['array1']
    city_id = data['array2']
    df = pd.DataFrame(tac, columns=['urban_core', 'urban_edge', 'rural_bgr'])
    df['ID'] = city_id
    df['urban_rural_tac_diff'] = df['urban_core'] - df['rural_bgr']
    return df


def filter_city_ids(df, city_ids):
    if city_ids is None:
        return df
    city_ids = set(np.asarray(city_ids, dtype=float))
    return df[df['ID'].astype(float).isin(city_ids)].copy()


def arid_city_ids(arid_shp, city_shp):
    arid = gpd.read_file(arid_shp)
    cities = gpd.read_file(city_shp)
    if cities.crs != arid.crs:
        cities = cities.to_crs(arid.crs)

    arid_geometry = arid.geometry.union_all() if hasattr(arid.geometry, 'union_all') else arid.geometry.unary_union
    cities_arid = cities[cities.geometry.intersects(arid_geometry)]
    return cities_arid['ID'].astype(float).to_numpy()


def panel_lag_tac_diff_scatter(ax, original_path, lag2_path, city_ids=None):
    df_original = filter_city_ids(load_tac_3zone_df(original_path), city_ids)
    df_lag2 = filter_city_ids(load_tac_3zone_df(lag2_path), city_ids)
    df_compare = df_original[['ID', 'urban_rural_tac_diff']].merge(
        df_lag2[['ID', 'urban_rural_tac_diff']],
        on='ID',
        suffixes=('_lag1', '_lag2')
    )
    df_compare = df_compare.dropna(subset=['urban_rural_tac_diff_lag1', 'urban_rural_tac_diff_lag2'])

    x = df_compare['urban_rural_tac_diff_lag1'].to_numpy()
    y = df_compare['urban_rural_tac_diff_lag2'].to_numpy()
    r, _ = st.pearsonr(x, y)
    rmse = np.sqrt(np.nanmean((y - x) ** 2))

    ax.scatter(x, y, s=14, color='#2166ac', alpha=0.65, edgecolors='none')
    ax.plot([-0.2, 0.2], [-0.2, 0.2], color='0.25', lw=0.9, ls='--', alpha=0.7)

    fit = np.poly1d(np.polyfit(x, y, 1))
    x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(x_fit, fit(x_fit), 'k-', alpha=0.7, lw=0.9)

    ax.set_xlim(-0.25, 0.15)
    ax.set_ylim(-0.08, 0.07)
    # ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('lag-1 ΔTAC')
    ax.set_ylabel('Lag-2 ΔTAC')
    ax.text(
        0.05, 0.95,
        f'r = {r:.2f}\nRMSE = {rmse:.3f}',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=12
    )
    return len(df_compare)


arid_ids = arid_city_ids(arid_shp, city_shp)
tac_data = np.load(current_dir + relative_path)
city_ids = tac_data['array2'].astype(float)
arid_mask = np.isin(city_ids, arid_ids)
TACs_mean = tac_data['array1'][arid_mask] # MODIS
tac_global_mean = np.nanmean(TACs_mean, axis=0)

fig, ax = plt.subplots(
    1, 4,
    figsize=(12.4 * 0.9, 3.2 * 0.8),
    gridspec_kw={'width_ratios': [1.5, 1, 1, 1]}
)

# Bar plot
bars = ax[0].bar(
    np.linspace(0, 2, 3),
    tac_global_mean,
    yerr=np.nanstd(TACs_mean, axis=0) * 0.25,
    width=0.42,
    color=['#2166ac', '#67a9cf', '#b2182b']
)

for bar in bars:
    bar.set_alpha(0.9)

ax[0].set_ylim([0.05, 0.10])
ax[0].set_xticks([0, 1, 2], ['UC', 'UE', 'RA'])
ax[0].set_ylabel('Lag-2 TAC')

# Create mask for valid (non-NaN) data points
mask_interface = ~np.isnan(TACs_mean[:, 0]) & ~np.isnan(TACs_mean[:, 1])
mask_rural = ~np.isnan(TACs_mean[:, 0]) & ~np.isnan(TACs_mean[:, 2])

# Urban core vs urban-rural interface plot
x_interface = TACs_mean[mask_interface, 0]
y_interface = TACs_mean[mask_interface, 1]

above_mask_interface = y_interface > x_interface
below_mask_interface = ~above_mask_interface

ax[1].plot(
    x_interface[above_mask_interface],
    y_interface[above_mask_interface],
    'o',
    mfc='none',
    color='C0'
)
ax[1].plot(
    x_interface[below_mask_interface],
    y_interface[below_mask_interface],
    'o',
    mfc='none',
    color='C3'
)

min_val = min(x_interface.min(), y_interface.min())
max_val = max(x_interface.max(), y_interface.max())
ax[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
z = np.polyfit(x_interface, y_interface, 1)
p = np.poly1d(z)
ax[1].plot(x_interface, p(x_interface), 'k-', alpha=0.7)

# Urban core vs rural background plot
x_rural = TACs_mean[mask_rural, 0]
y_rural = TACs_mean[mask_rural, 2]

above_mask_rural = y_rural > x_rural
below_mask_rural = ~above_mask_rural

ax[2].plot(
    x_rural[above_mask_rural],
    y_rural[above_mask_rural],
    'o',
    mfc='none',
    color='C0'
)
ax[2].plot(
    x_rural[below_mask_rural],
    y_rural[below_mask_rural],
    'o',
    mfc='none',
    color='C3'
)

min_val = min(x_rural.min(), y_rural.min())
max_val = max(x_rural.max(), y_rural.max())
ax[2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
z = np.polyfit(x_rural, y_rural, 1)
p = np.poly1d(z)
ax[2].plot(x_rural, p(x_rural), 'k-', alpha=0.7)

ax[1].set_xlabel('Lag-2 TAC$_{UC}$')
ax[1].set_ylabel('Lag-2 TAC$_{UE}$')
ax[1].set_xlim([-0.02, 0.32])
ax[1].set_ylim([-0.02, 0.32])
ax[2].set_xlabel('Lag-2 TAC$_{UC}$')
ax[2].set_ylabel('Lag-2 TAC$_{RA}$')
ax[2].set_xlim([-0.02, 0.32])
ax[2].set_ylim([-0.02, 0.32])

ax[0].title.set_text('Arid cities')
ax[1].title.set_text('Arid cities')
ax[2].title.set_text('Arid cities')
ax[3].title.set_text('Arid cities')

valid_compare = panel_lag_tac_diff_scatter(
    ax[3],
    current_dir + original_relative_path,
    current_dir + relative_path,
    arid_ids
)

print(f"Number of arid cities in shapefile: {len(arid_ids)}")
print(f"Number of arid cities in lag-2 TAC file: {np.sum(arid_mask)}")
print(f"Number of valid points (interface): {np.sum(mask_interface)}")
print(f"Number of valid points (rural): {np.sum(mask_rural)}")
print(f"Number of valid points (lag-1 vs lag-2 diff): {valid_compare}")

figToPath = current_dir + '/4_Figures/FigS01_tac_patterns_regression_MODIS_lag-2_arid'
fig.tight_layout()
fig.savefig(figToPath, dpi=900)

TACs_clean = TACs_mean[~np.isnan(TACs_mean).any(axis=1)]
print(st.ttest_rel(TACs_clean[:, 0], TACs_clean[:, 1]))
print(f"{st.linregress(x_interface, y_interface)}")
print(f"{st.linregress(x_rural, y_rural)}")
print(np.sum((y_interface-x_interface)>0)/len(x_interface))
print(np.sum((y_rural-x_rural)>0)/len(x_rural))
