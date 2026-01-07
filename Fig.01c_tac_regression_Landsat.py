import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import os
import numpy as np
import seaborn as sns

plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
plt.close()

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
relative_path = '/2_Output/tac_landsat_city_3zones_all.npz'
relative_path2 = '/2_Output/tac_nadir_city_3zones_all.npz'

# Calculate means and remove NaN values
TACs_mean = np.load(current_dir+relative_path)['array1']

# Create mask for valid (non-NaN) data points
mask_interface = ~np.isnan(TACs_mean[:,0]) & ~np.isnan(TACs_mean[:,1])
mask_rural = ~np.isnan(TACs_mean[:,0]) & ~np.isnan(TACs_mean[:,2])

fig, ax = plt.subplots(1,2, figsize=(6*0.7, 3.2*0.7))
# For urban core vs urban-rural interface plot
x_interface = TACs_mean[mask_interface,0]
y_interface = TACs_mean[mask_interface,1]

# Separate points above and below 1:1 line
above_mask_interface = y_interface > x_interface
below_mask_interface = ~above_mask_interface

# Plot points with different colors
ax[0].plot(x_interface[above_mask_interface], y_interface[above_mask_interface], 
           'o', mfc='none', color='C0')
ax[0].plot(x_interface[below_mask_interface], y_interface[below_mask_interface], 
           'o', mfc='none', color='C3')

# Add 1:1 and fit lines
min_val = min(x_interface.min(), y_interface.min())
max_val = max(x_interface.max(), y_interface.max())
ax[0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
z = np.polyfit(x_interface, y_interface, 1)
p = np.poly1d(z)
ax[0].plot(x_interface, p(x_interface), 'k-', alpha=0.7)

# For urban core vs rural background plot
x_rural = TACs_mean[mask_rural,0]
y_rural = TACs_mean[mask_rural,2]

# Separate points above and below 1:1 line
above_mask_rural = y_rural > x_rural
below_mask_rural = ~above_mask_rural

# Plot points with different colors
ax[1].plot(x_rural[above_mask_rural], y_rural[above_mask_rural], 
           'o', mfc='none', color='C0')
ax[1].plot(x_rural[below_mask_rural], y_rural[below_mask_rural], 
           'o', mfc='none', color='C3')

# Add 1:1 and fit lines
min_val = min(x_rural.min(), y_rural.min())
max_val = max(x_rural.max(), y_rural.max())
ax[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
z = np.polyfit(x_rural, y_rural, 1)
p = np.poly1d(z)
ax[1].plot(x_rural, p(x_rural), 'k-', alpha=0.7)

ax[0].set_xlabel('TAC$_{UC}$')
ax[0].set_ylabel('TAC$_{UE}$')
ax[0].set_xlim([-0.02,0.52])
ax[0].set_ylim([-0.02,0.52])
ax[1].set_xlabel('TAC$_{UC}$')
ax[1].set_ylabel('TAC$_{RA}$')
ax[1].set_xlim([-0.02,0.52])
ax[1].set_ylim([-0.02,0.52])
# Print number of valid points
print(f"Number of valid points (interface): {np.sum(mask_interface)}")
print(f"Number of valid points (rural): {np.sum(mask_rural)}")

figToPath = current_dir + '/4_Figures/FigS01_tac_regression_Landsat'
fig.tight_layout()
fig.savefig(figToPath, dpi=900)

import scipy.stats as st
print(f"{st.linregress(x_interface, y_interface)}")
print(f"{st.linregress(x_rural, y_rural)}")
print(np.sum((y_interface-x_interface)>0)/len(x_interface))
print(np.sum((y_rural-x_rural)>0)/len(x_rural))