# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tifffile as tf
#import matplotlib; matplotlib.use('Qt5Agg')
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
    'background_dots': '#e6e6e6' # Background dots color
}

def load_data():
    """Load and merge all required datasets"""
    # Load base datasets
    co2 = pd.read_csv(os.path.join('../..', '2_Output', 'city_CO2_trend.csv'))
    urban_irri = pd.read_csv(os.path.join('../..', '2_Output', 'vi_trends.csv'))
    gdp = pd.read_csv(os.path.join('../..', '2_Output', 'gdp_trends.csv'))
    pop = pd.read_csv(os.path.join('../..', '2_Output', 'pop_trends.csv'))
    uhi = pd.read_csv(current_dir + '/2_Output/uhi_trends.csv')
    AI = pd.read_csv(current_dir + '/2_Output/AI_trends.csv')
    sw = pd.read_csv(current_dir + '/2_Output/sw_trends.csv')
    ch = pd.read_csv(current_dir + '/2_Output/drivers/canopy_height.csv')
    df_tac = pd.read_csv(current_dir + '/2_Output/tac_trends.csv')

    return co2, urban_irri, gdp, pop, uhi, AI, sw, ch, df_tac

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
        
    for i, (correlations, p_values, title) in enumerate(zip(correlations_list, p_values_list, titles)):
        bars = axs[i].bar(range(len(correlations)),
                         list(correlations.values()),
                         color=COLOR_PALETTE['bar_color'],
                         edgecolor=COLOR_PALETTE['scatter_edge'],
                         alpha=0.8)
        
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
                y_pos = height + 0.00 if height >= 0 else height - 0.05
                axs[i].text(j, y_pos, marker, ha='center')
        
        axs[i].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        if i == len(correlations_list) - 1:
            plt.xticks(range(len(correlations)),
                      list(correlations.keys()),
                      rotation=45,
                      ha='right')

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')

# Load all data
co2, urban_irri, gdp, pop, uhi, AI, sw, ch, df_tac = load_data()

df_tac = pd.merge(df_tac, co2, on='ID')
df_tac = pd.merge(df_tac, urban_irri, on='ID')
df_tac = pd.merge(df_tac, gdp, on='ID')
df_tac = pd.merge(df_tac, pop, on='ID')
df_tac = pd.merge(df_tac, uhi, on='ID')
df_tac = pd.merge(df_tac, AI,on='ID')
df_tac = pd.merge(df_tac, sw, on='ID')
df_tac = pd.merge(df_tac, ch,on='ID')

df_tac['ntl_diff'] = df_tac['gpd_trend_diff']
df_tac['pop_diff'] = df_tac['pop_trend_diff']

df_tac['ch_diff'] = df_tac['ch_core'] - df_tac['ch_bg']
df_tac['tac_diff'] = df_tac['urban_rural_diff']
df_tac['co2_diff'] = df_tac['Urban_Core_CO2_trend'] - df_tac['Urban_Outedge_CO2_trend']
df_tac['urban_irri'] = df_tac['vi_trend_diff']
df_tac['SW'] = df_tac['sw_trend']
df_tac['AI'] = df_tac['AI_trend']
df_tac['uhi'] = df_tac['uhi_trend']
# climate: Ta, Pr, Rad, VPD
# urbanization factors: UHI, dCO2, urban irrigation, delta_vegC, delta_light, delta_PM2.5, vegetation type difference.
fig, axs = plt.subplots(1, 1, figsize=(5.5*0.8, 3.69 * 0.72*0.8),sharex=True)

# Prepare datasets for analysis
variable_sets = [
    ['SW', 'AI', 'urban_irri', 'ntl_diff', 'pop_diff', 'uhi', 'co2_diff',  'ch_diff', 'tac_diff'],
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
    # df_xgb.loc[df_xgb['urban_irri'] > 500,'urban_irri'] = 500
    correlations, p_values = calculate_partial_correlations(df_xgb, target)
    correlations_list.append(correlations)
    p_values_list.append(p_values)

# Create plots
plot_partial_correlations(axs, correlations_list, p_values_list, titles)

# Finalize and save plot
plt.xticks(range(8),['SW_t', 'AI_t', 'UI_t', 'ΔGDP_t', 'ΔPOP_t', 'UHI_t', 'ΔCO2_t',  'ΔHc'], ha='right')

plt.ylim([-0.2,0.1])
plt.tight_layout()
plt.savefig(os.path.join('../..', '4_Figures', 'partial_correlation_bars_temporal.png'),
            dpi=600, bbox_inches='tight')
# plt.show()

# Get top 4 important features
plt.figure(figsize=(6.2*0.8, 4*0.8))
X = df_tac[variable_sets[0]]

for i in range(4):
    plt.subplot(2, 2, i+1)
    
    # Determine variables
    val_label = ['urban_irri', 'uhi', 'co2_diff', 'SW']
    col_label = ['AI', 'AI', 'AI', 'urban_irri']
    labels = ['UI_t', 'UHI_t', 'ΔCO2_t', 'SW_t']
    
    # Get current feature values and SHAP values
    x_values = X[val_label[i]]
    y_values = df_tac['tac_diff']
    
    # Background scatter
    plt.plot(x_values, y_values, 'o', 
            alpha=0.3,
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
    plt.ylabel('ΔTAC_t' if i in [0,2] else '', fontsize=10)
    plt.ylim([-0.012, 0.014])
    
    # Customize colorbar
    cbar = plt.colorbar(scatter)
    cbar.ax.tick_params(labelsize=8)
    cbar.formatter = ticker.FormatStrFormatter('%.2f')
    cbar.update_ticks()
    
plt.tight_layout()
plt.savefig(os.path.join('../..', '4_Figures', 'scatter_landsat_temporal_temp.png'),
            dpi=900, bbox_inches='tight')
plt.show()

