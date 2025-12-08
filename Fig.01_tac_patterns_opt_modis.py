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

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
relative_path = '/2_Output/tac_nadir_city_3zones_all.npz'
relative_path2 = '/2_Output/tac_landsat_city_3zones_all.npz'

TACs_mean = np.load(current_dir+relative_path2)['array1'] # Landsat
tac_global_mean = np.nanmean(TACs_mean, axis=0)

# Modify bar plot to add dots pattern
fig, ax = plt.subplots(1,1,figsize=(4.3*0.9,2.1*0.9))
bars = ax.bar(np.linspace(0,2,3), tac_global_mean,
              yerr=np.nanstd(TACs_mean,axis=0)*0.25,
              width=0.30,
              color=['#2166ac', '#67a9cf', '#b2182b'])

# Add dots pattern to bars
for bar in bars:
    bar.set_alpha(0.9)  # Slightly reduce alpha to make pattern more visible

ax.set_ylim([0.10,0.22])
ax.set_xticks([0,1,2], ['UC','UE','RA'])
# ax.set_yticks([0.15,0.17,0.19,0.21])
figToPath = current_dir + '/4_Figures/Fig01a_tac_3zones_bar_Landsat'
fig.tight_layout()
fig.savefig(figToPath, dpi=900)


TACs_mean =  np.load(current_dir+relative_path)['array1'] # Landsat
tac_global_mean = np.nanmean(TACs_mean, axis=0)


# Modify bar plot to add dots pattern
fig, ax = plt.subplots(1,1,figsize=(4.3*0.9,2.1*0.9))
bars = ax.bar(np.linspace(0,2,3), tac_global_mean, 
              yerr=np.nanstd(TACs_mean,axis=0)*0.25, 
              width=0.30, 
              color=['#2166ac', '#67a9cf', '#b2182b'])

# Add dots pattern to bars
for bar in bars:
    bar.set_hatch('...')
    bar.set_alpha(0.9)  # Slightly reduce alpha to make pattern more visible

ax.set_ylim([0.10,0.22])
ax.set_xticks([0,1,2], ['UC','UE','RA'])
# ax.set_yticks([0.15,0.17,0.19,0.21])
figToPath = current_dir + '/4_Figures/Fig01a_tac_3zones_bar_MODIS'
fig.tight_layout()
fig.savefig(figToPath, dpi=900)

import scipy.stats as st
TACs_mean=TACs_mean[~np.isnan(TACs_mean).any(axis=1)]
st.ttest_rel(TACs_mean[:,0], TACs_mean[:,1])