# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
plt.rc('lines', linewidth=1.05)  # Increase default line width
plt.rc('axes', linewidth=1.05)   # Increase axes line width
plt.rc('grid', linewidth=1.05)   # Increase grid line width
plt.tick_params(width=1.05, labelsize=14)
plt.rc('xtick.major', size=4, width=1.05)    # Increase length and width of major ticks
plt.rc('ytick.major', size=4, width=1.05)    # Increase length and width of major ticks
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
    'main_blue': '#2166ac',      # Primary blue
    'light_blue': '#67a9cf',     # Light blue
    'main_red': '#b2182b',       # Primary red
    'bar_color': '#4393c3',      # Bar color
    'grid_color': '#cccccc',     # Grid lines
    'fit_line': '#404040',       # Fit line color
    'scatter_edge': '#333333',   # Scatter plot edge color
    'background_dots': '#bababa' # Background dots color
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

def calculate_partial_correlations(df_xgb, target_col):
    """Calculate partial correlations and p-values for normalized variables"""
    # Prepare and normalize data
    X = df_xgb.drop(target_col, axis=1)
    y = df_xgb[target_col]
    X_normalized = (X - X.mean()) / X.std()
    
    # Calculate partial correlations
    partial_correlations = {}
    p_values = {}
    for col in X_normalized.columns:
        controlling_vars = [c for c in X_normalized.columns if c != col]
        pcorr = pg.partial_corr(data=pd.concat([X_normalized, y], axis=1),
                               x=col,
                               y=target_col,
                               covar=controlling_vars)
        partial_correlations[col] = pcorr['r'].values[0]
        p_values[col] = pcorr['p-val'].values[0]
    
    return partial_correlations, p_values

def plot_partial_correlations(axs, correlations_list, p_values_list, titles):
    """Create bar plots for partial correlations with significance markers"""
    # Convert single axes to list for consistent handling
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    strs=['//','//','...','...','...','...','...','\\\\','\\\\']
    for i, (correlations, p_values, title,str) in enumerate(zip(correlations_list, p_values_list, titles,strs)):
        bars = axs[i].bar(range(len(correlations)),
                         list(correlations.values()),
                         color=COLOR_PALETTE['bar_color'],
                         edgecolor=COLOR_PALETTE['scatter_edge'],
                         alpha=0.8,hatch=strs)
        print(i)
        
        # Add significance markers
        for j, p in enumerate(p_values.values()):
            height = list(correlations.values())[j]
            marker = ''
            if p < 0.001:
                marker = '***'
            elif p < 0.01:
                marker = '**'
            elif p < 0.05:
                marker = '*'
            
            if marker:
                # Adjust y position based on whether correlation is positive or negative
                y_pos = height + 0.00 if height >= 0 else height - 0.08
                axs[i].text(j, y_pos, marker, ha='center')
        
        axs[i].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        if i == len(correlations_list) - 1:
            plt.xticks(range(len(correlations)),
                      list(correlations.keys()),
                      rotation=45,
                      ha='right')

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
relative_path = '/2_Output/tac_nadir_city_3zones.npz'
relative_path2 = '/2_Output/tac_landsat_city_3zones.npz'

TACs = np.load(current_dir+relative_path2)['array1'] # LANDSAT
ID = np.load(current_dir+relative_path2)['array2'] # MODIS
ID2 = np.load(current_dir+relative_path)['array2']
TACs_mean = np.nanmean(TACs, axis=2)
tac_global_mean = np.nanmean(TACs_mean, axis=0)

df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_edge','rural_bgr']
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
df_tac = pd.merge(df_tac, ntl_vegc_ndvi,on='ID')
df_tac = pd.merge(df_tac, pop,on='ID')
df_tac = pd.merge(df_tac, shann,on='ID')

df_tac['ntl_diff'] = df_tac['ntl_core_all'] - df_tac['ntl_bg_all']
df_tac['pop_diff'] = df_tac['pop_core_all'] - df_tac['pop_bg_all']
df_tac['shann_diff'] = df_tac['shann_core_all'] - df_tac['shann_bg_all']

df_tac['ch_diff'] = df_tac['ch_core'] - df_tac['ch_bg']
df_tac['ndvi_diff'] = df_tac['ndvi_core'] - df_tac['ndvi_bg']
df_tac['tac_diff'] = df_tac['urban_core'] - df_tac['rural_bgr']
df_tac['co2_diff'] = df_tac['Urban_Core_CO2'] - df_tac['Urban_Outedge_CO2']+0.5

df_tac['LE_core'] = df_tac['Rn_core'] + df_tac['Q_core'] - df_tac['H_core'] - df_tac['Rn_core'] * 0.25
df_tac['LE_bg'] = df_tac['Rn_bg'] + df_tac['Q_bg'] - df_tac['H_bg'] - df_tac['Rn_bg'] * 0.25
df_tac['urban_irri'] = df_tac['LE_core']-df_tac['LE_bg']
df_tac['Q_diff'] = df_tac['Q_core'] - df_tac['Q_bg']
df_tac['delta_Tc'] = df_tac['Tc_core'] - df_tac['Tc_bg']
# climate: Ta, Pr, Rad, VPD
# urbanization factors: UHI, dCO2, urban irrigation, delta_vegC, delta_light, delta_PM2.5, vegetation type difference.
fig, axs = plt.subplots(1, 1, figsize=(5.5*0.8, 3.5 * 0.72*0.8),sharex=True)

# Prepare datasets for analysis
variable_sets = [
    ['SW', 'AI', 'urban_irri', 'ntl_diff', 'pop_diff', 'uhi', 'co2_diff',  'ch_diff', 'shann_diff','tac_diff'],
    #['SW', 'AI', 'LE_core', 'ntl_core_all', 'pop_core_all', 'Tc_core', 'co2_diff', 'ch_core', 'urban_core'],
    #['SW', 'AI', 'LE_bg', 'ntl_bg_all', 'pop_bg_all', 'Tc_bg', 'co2_diff',  'ch_bg', 'rural_bgr']
]

target_cols = ['tac_diff', 'urban_core', 'rural_bgr']
titles = ['TAC Difference']

# Calculate correlations for each set
correlations_list = []
p_values_list = []
for vars, target in zip(variable_sets, target_cols):
    df_xgb = df_tac[vars].dropna()
    df_xgb.loc[df_xgb['urban_irri'] > 500,'urban_irri'] = 500
    correlations, p_values = calculate_partial_correlations(df_xgb, target)
    correlations_list.append(correlations)
    p_values_list.append(p_values)

# Create plots
plot_partial_correlations(axs, correlations_list, p_values_list, titles)

# Finalize and save plot
plt.ylim([-0.36,0.32])
plt.xticks(range(9),['SW', 'AI', 'UI', 'ΔGDP', 'ΔPOP', 'UHI', 'ΔCO2',  'ΔHc', 'ΔHet'], ha='right')
plt.tight_layout()
plt.savefig(os.path.join('..', '4_Figures', 'partial_correlation_bars.png'), 
            dpi=600, bbox_inches='tight')
# plt.show()


# Get top 4 important features
plt.figure(figsize=(6.2*0.8, 4*0.8))
X = df_tac[variable_sets[0]]
X.loc[X['urban_irri'] > 500,'urban_irri'] = 500
X.loc[X['ntl_diff'] > 100, 'ntl_diff'] = 100

for i in range(4):
    plt.subplot(2, 2, i+1)
    
    # Determine variables
    val_label = ['urban_irri', 'AI', 'ch_diff', 'ntl_diff', ]
    col_label = ['AI', 'urban_irri', 'AI', 'AI']
    labels = ['UI', 'AI', 'ΔHc',  'ΔGDP']
    
    # Get current feature values and SHAP values
    x_values = X[val_label[i]]
    y_values = df_tac['tac_diff']
    
    # Background scatter
    plt.plot(x_values, y_values, 'o', 
            alpha=0.5,
            color=COLOR_PALETTE['background_dots'],
            markersize=3)
    
    # Add linear fit line
    mask = ~np.isnan(x_values) & ~np.isnan(y_values)
    slope, intercept, r_value, p_value, _ = st.linregress(x_values[mask], y_values[mask])
    x_fit = np.array([x_values.min(), x_values.max()])
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, '-', lw=1, color='grey', alpha=0.8)
    print(r_value)
    
    # Calculate binned means for smooth line
    bins = np.linspace(x_values.min(), x_values.max(), 21)  # 20 bins + 1 edge
    digitized = np.digitize(x_values, bins)
    bin_means = [y_values[digitized == j].mean() for j in range(1, len(bins))]
    bin_means[-1] = y_values[x_values==x_values.max()].mean()
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Fill NaN values with interpolation
    bin_means = pd.Series(bin_means).interpolate(method='linear').values
    
    clr_means = [X[col_label[i]][digitized == j].mean() for j in range(1, len(bins))]
    clr_means[-1] = X[col_label[i]][x_values==x_values.max()].values[0]
    
    # Fill NaN values with interpolation
    bin_means = pd.Series(bin_means).interpolate(method='linear').values 
    
    # Binned scatter plot with custom colormap
    custom_cmap = plt.cm.RdYlBu_r  # Red-Yellow-Blue reversed colormap
    scatter = plt.scatter(bin_centers,
                         bin_means,
                         c=clr_means,
                         cmap=custom_cmap,
                         s=25,
                         edgecolor=COLOR_PALETTE['scatter_edge'],
                         linewidth=0.5,
                         alpha=0.9,
                         zorder=3)
    
    # Fit line
    plt.plot(x_fit, y_fit, '-', 
            lw=1.5, 
            color=COLOR_PALETTE['fit_line'],
            alpha=0.8)
    
    # Customize plot
    plt.grid(True, linestyle='--', alpha=0.2, color=COLOR_PALETTE['grid_color'])
    plt.xlabel(labels[i], fontsize=10)
    plt.ylabel('ΔTAC' if i in [0,2] else '', fontsize=10)
    plt.ylim([-0.2, 0.07])
    
    # Customize colorbar
    cbar = plt.colorbar(scatter)
    cbar.ax.tick_params(labelsize=8)
    cbar.formatter = ticker.FormatStrFormatter('%.2f')
    cbar.update_ticks()
    
plt.tight_layout()
plt.savefig(os.path.join('..', '4_Figures', 'scatter_landsat.png'),
            dpi=900, bbox_inches='tight')
plt.show()