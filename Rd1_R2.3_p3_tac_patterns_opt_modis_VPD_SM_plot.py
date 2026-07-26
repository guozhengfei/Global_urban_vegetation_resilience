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
relative_path = '/2_Output/tac_nadir_city_3zones_all_vpd_soil.npz'

TACs_mean = np.load(current_dir+relative_path)['array1'] # MODIS
tac_global_mean = np.nanmean(TACs_mean, axis=0)

fig, ax = plt.subplots(
    1, 3,
    figsize=(9.2 * 0.9, 3.2 * 0.8),
    gridspec_kw={'width_ratios': [1.8, 1, 1]}
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

ax[0].set_ylim([0.10, 0.22])
ax[0].set_xticks([0, 1, 2], ['UC', 'UE', 'RA'])
ax[0].set_ylabel('TAC')

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

ax[1].set_xlabel('TAC$_{UC}$')
ax[1].set_ylabel('TAC$_{UE}$')
ax[1].set_xlim([-0.02, 0.52])
ax[1].set_ylim([-0.02, 0.52])
ax[2].set_xlabel('TAC$_{UC}$')
ax[2].set_ylabel('TAC$_{RA}$')
ax[2].set_xlim([-0.02, 0.52])
ax[2].set_ylim([-0.02, 0.52])

print(f"Number of valid points (interface): {np.sum(mask_interface)}")
print(f"Number of valid points (rural): {np.sum(mask_rural)}")

figToPath = current_dir + '/4_Figures/FigS01_tac_patterns_regression_MODIS_vpd_sm'
fig.tight_layout()
fig.savefig(figToPath, dpi=900)

TACs_clean = TACs_mean[~np.isnan(TACs_mean).any(axis=1)]
print(st.ttest_rel(TACs_clean[:, 0], TACs_clean[:, 1]))
print(f"{st.linregress(x_interface, y_interface)}")
print(f"{st.linregress(x_rural, y_rural)}")
print(np.sum((y_interface-x_interface)>0)/len(x_interface))
print(np.sum((y_rural-x_rural)>0)/len(x_rural))
