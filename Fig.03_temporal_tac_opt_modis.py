import geopandas as gpd
import pandas as pd
import matplotlib; matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
plt.close()
import os
import numpy as np
import scipy.stats as st

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
TACs = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array1']
TACs = TACs[:,:,2:-1]
ID = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array2']
ID2 = np.load(current_dir+'/2_Output/tac_landsat_city_3zones.npz')['array2']

TAC_core = TACs[:,0,:] # 711 rows, 18 columns
TAC_rural = TACs[:,-1]
TAC_diff =  TAC_core - TAC_rural

# Calculate median and 0.5 standard deviation for each time point
medians = np.nanmedian(TAC_diff, axis=0)
std_devs = np.nanstd(TAC_diff, axis=0) * 0.25

# fig, ax = plt.subplots(3, 1, figsize=(4*0.7, 7*0.7),sharex=True)
fig, ax = plt.subplots(1, 3, figsize=(10.5*0.8, 2.6*0.8))

# Plot the first panel with error bars
ax[0].errorbar(range(1, 19), medians, yerr=std_devs, fmt='o', mfc='None', color='#66c2a5')
ax[0].set_ylabel('ΔTAC')
ax[0].set_xticks([3, 8, 13, 18])
ax[0].set_xticklabels([2005, 2010, 2015, 2020])

# Add linear fit line for the first panel
slope, intercept, r_value, p_value, std_err = st.linregress(range(1, 19), medians)
ax[0].plot(range(1, 19), intercept + slope * np.array(range(1, 19)), 'k-', label=f'Slope={slope:.1e}')
ax[0].legend()

medians1 = np.nanmedian(TAC_core, axis=0)
std_devs1 = np.nanstd(TAC_core, axis=0) * 0.25
# Plot the second panel with error bars
ax[2].errorbar(range(1, 19), medians1, yerr=std_devs1, fmt='o', mfc='None', color='#fc8d62')
ax[2].set_ylabel(r"TAC$_{UC}$")
ax[2].set_xticks([3, 8, 13, 18])
ax[2].set_xticklabels([2005, 2010, 2015, 2020])

# Add linear fit line for the second panel
slope1, intercept1, r_value1, p_value1, std_err1 = st.linregress(range(1, 19), medians1)
ax[2].plot(range(1, 19), intercept1 + slope1 * np.array(range(1, 19)), 'k-', label=f'Slope={slope1:.1e}')
ax[2].legend()

medians2 = np.nanmedian(TAC_rural, axis=0)
std_devs2 = np.nanstd(TAC_rural, axis=0) * 0.25
# Plot the third panel with error bars
ax[1].errorbar(range(1, 19), medians2, yerr=std_devs2, fmt='o', mfc='None', color='#8da0cb')
ax[1].set_xlabel('Year')
ax[1].set_ylabel(r"TAC$_{RU}$")
ax[1].set_xticks([3, 8, 13, 18])
ax[1].set_xticklabels([2005, 2010, 2015, 2020])

# Add linear fit line for the third panel
slope2, intercept2, r_value2, p_value2, std_err2 = st.linregress(range(1, 19), medians2)
ax[1].plot(range(1, 19), intercept2 + slope2 * np.array(range(1, 19)), 'k-', label=f'Slope={slope2:.1e}')
ax[1].legend()

# Save the figure
figToPath = current_dir + '/4_Figures/Fig03_temporal_tac_opt_modis'
fig.tight_layout()
fig.savefig(figToPath, dpi=900)
# plt.close(fig)


