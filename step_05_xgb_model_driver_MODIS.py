# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tifffile as tf
# import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

plt.rc('font', family='Arial')
plt.rc('lines', linewidth=1.05)  # Increase default line width
plt.rc('axes', linewidth=1.05)  # Increase axes line width
plt.rc('grid', linewidth=1.05)  # Increase grid line width
plt.tick_params(width=1.05, labelsize=14)
plt.rc('xtick.major', size=4, width=1.05)  # Increase length and width of major ticks
plt.rc('ytick.major', size=4, width=1.05)  # Increase length and width of major ticks
plt.close()
import matplotlib.ticker as ticker
import scipy.stats as st
import warnings

warnings.filterwarnings("ignore")
import os
import geopandas as gpd
import seaborn as sns
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
import pingouin as pg

# Define a consistent color palette
COLOR_PALETTE = {
    'main_blue': '#2166ac',  # Primary blue
    'light_blue': '#67a9cf',  # Light blue
    'main_red': '#b2182b',  # Primary red
    'bar_color': '#4393c3',  # Bar color
    'grid_color': '#cccccc',  # Grid lines
    'fit_line': '#404040',  # Fit line color
    'scatter_edge': '#333333',  # Scatter plot edge color
    'background_dots': '#bababa'  # Background dots color
}


def load_data():
    """Load and merge all required datasets"""
    # Load base datasets
    urban_factors = pd.read_csv(os.path.join('..', '2_Output', 'drivers', 'urban_factors_effect_v2.csv'))
    co2 = pd.read_csv(os.path.join('..', '2_Output', 'city_mean_CO2.csv'))
    vpd = pd.read_csv(os.path.join('..', '..', 'urban_env_data', 'vpd_csv_urban_751.csv'))
    ntl_vegc_ndvi = pd.read_csv(os.path.join('..', '2_Output', 'drivers', 'ntl_ndvi_veg.csv'))
    pop = pd.read_csv(os.path.join('..', '2_Output', 'drivers', 'pop.csv'))
    shann = pd.read_csv(os.path.join('..', '2_Output', 'drivers', 'shann.csv'))
    climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
    AI = pd.read_csv(current_dir + '/2_Output/AI_csv.csv')
    ch = pd.read_csv(current_dir + '/2_Output/drivers/canopy_height.csv')

    # Process climate data
    climate['MAT'] = climate['MAT'] - 273.15
    climate['MAP'] = climate['MAP'] * 24 * 1000

    return urban_factors, co2, vpd, ntl_vegc_ndvi, pop, shann, climate, AI, ch


current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
relative_path = '/2_Output/tac_nadir_city_3zones.npz'
relative_path2 = '/2_Output/tac_landsat_city_3zones.npz'

TACs = np.load(current_dir + relative_path2)['array1']  # LANDSAT
ID = np.load(current_dir + relative_path2)['array2']  # MODIS
ID2 = np.load(current_dir + relative_path)['array2']
TACs_mean = np.nanmean(TACs, axis=2)
tac_global_mean = np.nanmean(TACs_mean, axis=0)

df_tac = pd.DataFrame(TACs_mean)
df_tac.columns = ['urban_core', 'urban_edge', 'rural_bgr']
df_tac['ID'] = ID

df_tac = df_tac[df_tac['ID'].isin(ID2)]

# Load all data
urban_factors, co2, vpd, ntl_vegc_ndvi, pop, shann, climate, AI, ch = load_data()

df_tac = pd.merge(df_tac, AI, on='ID')
df_tac = pd.merge(df_tac, ch, on='ID')
df_tac = pd.merge(df_tac, climate, on='ID')
df_tac = pd.merge(df_tac, urban_factors, on='ID')
df_tac = pd.merge(df_tac, co2, on='ID')
df_tac = pd.merge(df_tac, vpd, on='ID')
df_tac = pd.merge(df_tac, ntl_vegc_ndvi, on='ID')
df_tac = pd.merge(df_tac, pop, on='ID')
df_tac = pd.merge(df_tac, shann, on='ID')

df_tac['ntl_diff'] = df_tac['ntl_core_all'] - df_tac['ntl_bg_all']
df_tac['pop_diff'] = df_tac['pop_core_all'] - df_tac['pop_bg_all']
df_tac['shann_diff'] = df_tac['shann_core_all'] - df_tac['shann_bg_all']

df_tac['ch_diff'] = df_tac['ch_core'] - df_tac['ch_bg']
df_tac['ndvi_diff'] = df_tac['ndvi_core'] - df_tac['ndvi_bg']
df_tac['tac_diff'] = df_tac['urban_core'] - df_tac['rural_bgr']
df_tac['co2_diff'] = df_tac['Urban_Core_CO2'] - df_tac['Urban_Outedge_CO2'] + 0.5

df_tac['LE_core'] = df_tac['Rn_core'] + df_tac['Q_core'] - df_tac['H_core'] - df_tac['Rn_core'] * 0.25
df_tac['LE_bg'] = df_tac['Rn_bg'] + df_tac['Q_bg'] - df_tac['H_bg'] - df_tac['Rn_bg'] * 0.25
df_tac['urban_irri'] = df_tac['LE_core'] - df_tac['LE_bg']
df_tac['Q_diff'] = df_tac['Q_core'] - df_tac['Q_bg']
df_tac['delta_Tc'] = df_tac['Tc_core'] - df_tac['Tc_bg']
# climate: Ta, Pr, Rad, VPD
# urbanization factors: UHI, dCO2, urban irrigation, delta_vegC, delta_light, delta_PM2.5, vegetation type difference.
fig, axs = plt.subplots(1, 1, figsize=(5.5 * 0.8, 3.5 * 0.72 * 0.8), sharex=True)

# Prepare datasets for analysis
variable_sets = [
    ['SW', 'AI', 'urban_irri', 'ntl_diff', 'pop_diff', 'uhi', 'co2_diff', 'ch_diff', 'shann_diff'],
    # ['SW', 'AI', 'LE_core', 'ntl_core_all', 'pop_core_all', 'Tc_core', 'co2_diff', 'ch_core', 'urban_core'],
    # ['SW', 'AI', 'LE_bg', 'ntl_bg_all', 'pop_bg_all', 'Tc_bg', 'co2_diff',  'ch_bg', 'rural_bgr']
]

df_xgb = df_tac[variable_sets[0]]#'vpd',
df_xgb.columns=['SW', 'AI', 'UI', 'ΔGDP', 'ΔPOP', 'UHI', 'ΔCO2',  'ΔHc', 'ΔHet']
# climate: Ta, Pr, Rad, VPD
# urbanization factors: UHI, dCO2, urban irrigation, delta_vegC, delta_light, delta_PM2.5, vegetation type difference.
df_xgb = df_xgb.dropna()
# Create correlation matrix
corr_matrix = df_xgb.corr()
corr_matrix.replace(1,np.nan,inplace=True)

# Create figure with specified size
plt.figure(figsize=(6, 6))

# Create heatmap
sns.heatmap(corr_matrix, 
            annot=True,          # Show correlation values
            cmap='RdBu_r',       # Red-Blue diverging colormap
            vmin=-1, vmax=1,     # Set correlation range
            center=0,            # Center colormap at 0
            fmt='.2f',           # Format correlation values to 2 decimal places
            square=True)         # Make cells square

# Customize plot
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join('..', '4_Figures', 'correlation_heatmap.png'), dpi=600, bbox_inches='tight')
plt.show()

np.nanmean(abs(corr_matrix))