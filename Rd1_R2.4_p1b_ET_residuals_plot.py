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
from sklearn.linear_model import LinearRegression
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
    et_diff = pd.read_csv(current_dir + '/2_Output/pure_veg/monthly_landsat_et_PM_global_urban_rural_diff_by_vegtype_100.csv')
    tac_diff_veg = pd.read_csv(current_dir + '/2_Output/VI_Landsat/tree_grass_dominant_urban_rural_tac_diff_by_city.csv')
    et_diff_cols = [
        'all_urban_rural_et_diff',
        'tree_dominant_urban_rural_et_diff',
        'grass_dominant_urban_rural_et_diff',
    ]
    et_diff[et_diff_cols] = et_diff[et_diff_cols] * 12
    
    # Process climate data
    climate['MAT'] = climate['MAT'] - 273.15
    climate['MAP'] = climate['MAP'] * 24 * 1000
    
    return urban_factors, co2, vpd, ntl_vegc_ndvi, pop, shann, climate, AI, ch, et_diff, tac_diff_veg

def calculate_partial_correlations(df_xgb, target_col):
    """Calculate partial correlations and p-values for normalized variables"""
    # Prepare and normalize data
    X = df_xgb.drop(target_col, axis=1)
    y = df_xgb[target_col]
    X_normalized = (X - X.mean()) / X.std()
    
    # Calculate partial correlations
    partial_correlations = {}
    p_values = {}
    i=0
    for col in X_normalized.columns:
        controlling_vars = [c for c in X_normalized.columns if c != col]
        pcorr = pg.partial_corr(data=pd.concat([X_normalized, y], axis=1),
                               x=col,
                               y=target_col,
                               covar=controlling_vars)
        # if i == 2:
        #     pcorr['r'].values[0] = pcorr['r'].values[0]-0.05
        partial_correlations[col] = pcorr['r'].values[0]
        p_values[col] = pcorr['p-val'].values[0]
        i = i + 1

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
TACs_mean = np.nanmean(TACs, axis=2) if TACs.ndim == 3 else TACs

df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_edge','rural_bgr']
df_tac['ID'] = ID

df_tac = df_tac[df_tac['ID'].isin(ID2)]

# Load all data
urban_factors, co2, vpd, ntl_vegc_ndvi, pop, shann, climate, AI, ch, et_diff, tac_diff_veg = load_data()

df_tac = pd.merge(df_tac, AI, on='ID')
df_tac = pd.merge(df_tac, ch, on='ID')
df_tac = pd.merge(df_tac, climate, on='ID')
df_tac = pd.merge(df_tac, urban_factors, on='ID')
df_tac = pd.merge(df_tac, co2, on='ID')
df_tac = pd.merge(df_tac, vpd, on='ID')
df_tac = pd.merge(df_tac, ntl_vegc_ndvi,on='ID')
df_tac = pd.merge(df_tac, pop,on='ID')
df_tac = pd.merge(df_tac, shann,on='ID')
df_tac = pd.merge(
    df_tac,
    et_diff[[
        'id',
        'all_urban_rural_et_diff',
        'tree_dominant_urban_rural_et_diff',
        'grass_dominant_urban_rural_et_diff',
    ]],
    left_on='ID',
    right_on='id'
)
df_tac = df_tac.drop(columns=['id'])
df_tac = pd.merge(
    df_tac,
    tac_diff_veg[[
        'ID',
        'all_urban_rural_tac_diff',
        'tree_urban_rural_tac_diff',
        'grass_urban_rural_tac_diff',
    ]],
    on='ID'
)

df_tac['ntl_diff'] = df_tac['ntl_core_all'] - df_tac['ntl_bg_all']
df_tac['pop_diff'] = df_tac['pop_core_all'] - df_tac['pop_bg_all']
df_tac['shann_diff'] = df_tac['shann_core_all'] - df_tac['shann_bg_all']

df_tac['ch_diff'] = df_tac['ch_core'] - df_tac['ch_bg']
df_tac['fvc_diff'] = df_tac['veg_core_all'] - df_tac['veg_bg_all']
df_tac['ndvi_diff'] = df_tac['ndvi_core_all'] - df_tac['ndvi_bg_all']
df_tac['tac_diff'] = df_tac['urban_core'] - df_tac['rural_bgr']
df_tac['co2_diff'] = df_tac['Urban_Core_CO2'] - df_tac['Urban_Outedge_CO2']+0.5
df_tac['Rn_diff'] = df_tac['Rn_core'] - df_tac['Rn_bg']
df_tac['SW_diff'] = df_tac['SW_core'] - df_tac['SW_bg']
df_tac['LW_diff'] = df_tac['LW_core'] - df_tac['LW_bg']
df_tac['Q_diff'] = df_tac['Q_core'] - df_tac['Q_bg']
required_surface_cols = ['albedo_core', 'albedo_bg', 'emissivity_core', 'emissivity_bg']
missing_surface_cols = [col for col in required_surface_cols if col not in df_tac.columns]
if missing_surface_cols:
    raise ValueError(
        'urban_factors_effect_v2.csv must include these columns: '
        + ', '.join(missing_surface_cols)
    )
df_tac['albedo_diff'] = df_tac['albedo_core'] - df_tac['albedo_bg']
df_tac['emissivity_diff'] = df_tac['emissivity_core'] - df_tac['emissivity_bg']

df_tac['LE_core'] = df_tac['Rn_core'] + df_tac['Q_core'] - df_tac['H_core'] - df_tac['Rn_core'] * 0.25
df_tac['LE_bg'] = df_tac['Rn_bg'] + df_tac['Q_bg'] - df_tac['H_bg'] - df_tac['Rn_bg'] * 0.25

df_tac['delta_Tc'] = df_tac['Tc_core'] - df_tac['Tc_bg']
# climate: Ta, Pr, Rad, VPD
# urbanization factors: UHI, dCO2, urban irrigation, delta_vegC, delta_light, delta_PM2.5, vegetation type difference.
group_configs = [
    ('All vegetation', 'all_urban_rural_et_diff', 'all_urban_rural_tac_diff')
]
requested_et_predictors = [
    # ('Rn_diff', 'ΔRn'),
    ('SW_diff', 'ΔSW'),
    ('LW_diff', 'ΔLW'),
    ('Q_diff', 'ΔQ'),
    ('albedo_diff', 'Δalbedo'),
    ('emissivity_diff', 'Δemissivity'),
    ('fvc_diff', 'ΔFVC'),
    ('ndvi_diff', 'ΔNDVI'),
    ('uhi', 'UHI'),
    # ('co2_diff', 'ΔCO2'),
    ('ch_diff', 'ΔHc')
]
et_residual_predictors = [col for col, label in requested_et_predictors if col in df_tac.columns]
et_residual_labels = [label for col, label in requested_et_predictors if col in df_tac.columns]
partial_corr_predictor_cols = [
    'SW',
    'AI',
    'urban_irri',
    'ntl_diff',
    'pop_diff',
    'uhi',
    'co2_diff',
    'ch_diff',
    'shann_diff',
]
partial_corr_xtick_labels = ['SW', 'AI', 'UI', 'ΔGDP', 'ΔPOP', 'UHI', 'ΔCO2', 'ΔHc', 'ΔHet']

et_model_cols = et_residual_predictors + ['all_urban_rural_et_diff']
df_et_model = df_tac[et_model_cols].dropna().copy()
et_model = LinearRegression()
et_model.fit(df_et_model[et_residual_predictors], df_et_model['all_urban_rural_et_diff'])
df_et_model['predicted_et_diff'] = et_model.predict(df_et_model[et_residual_predictors])
df_et_model['et_residual'] = df_et_model['all_urban_rural_et_diff'] - df_et_model['predicted_et_diff']
df_tac['all_urban_rural_et_residual'] = np.nan
df_tac.loc[df_et_model.index, 'all_urban_rural_et_residual'] = df_et_model['et_residual']
print('Residual ΔET model R2', r2_score(
    df_et_model['all_urban_rural_et_diff'],
    df_et_model['predicted_et_diff']
))


selected_residual_items = [
    ('predicted_et_diff', 'Linear model performance'),
    # ('Rn_diff', 'ΔRn (W/m²)'),
    ('SW_diff', 'ΔSW (W/m²)'),
    ('LW_diff', 'ΔLW (W/m²)'),
    ('Q_diff', 'ΔQ (W/m²)'),
    ('albedo_diff', 'Δalbedo'),
    ('emissivity_diff', 'Δemissivity'),
    ('uhi', 'UHI'),
    # ('co2_diff', 'ΔCO2'),
    ('ndvi_diff', 'ΔNDVI'),
    ('ch_diff', 'ΔHc'),
    ('fvc_diff', 'ΔFVC'),
]
fig_selected, axs_selected = plt.subplots(2, 5, figsize=(13.2*0.9, 6.0*0.8))
axs_selected = axs_selected.ravel()

for i, (col, label) in enumerate(selected_residual_items):
    ax = axs_selected[i]
    if col == 'predicted_et_diff':
        x_values = df_et_model['all_urban_rural_et_diff']
        y_values = df_et_model['predicted_et_diff']
        x_label = 'Original ΔET (mm/year)'
        y_label = 'Predicted ΔET (mm/year)'
        title = label
    else:
        x_values = df_et_model[col]
        y_values = df_et_model['et_residual']
        x_label = label
        y_label = 'Residual ΔET (mm/year)'
        title = f'Residual ΔET vs {label}'

    mask = ~np.isnan(x_values) & ~np.isnan(y_values)
    ax.plot(
        x_values[mask],
        y_values[mask],
        'o',
        alpha=0.5,
        color=COLOR_PALETTE['background_dots'],
        markersize=3
    )
    if np.sum(mask) > 2:
        if col == 'predicted_et_diff':
            xy_min = min(x_values[mask].min(), y_values[mask].min())
            xy_max = max(x_values[mask].max(), y_values[mask].max())
            ax.plot([xy_min, xy_max], [xy_min, xy_max], '--', lw=1.0, color='0.45')
            r_value = st.pearsonr(x_values[mask], y_values[mask])[0]
            text_label = f'r = {r_value:.2f}\nR² = {r2_score(x_values[mask], y_values[mask]):.2f}'
        else:
            slope, intercept, r_value, p_value, _ = st.linregress(x_values[mask], y_values[mask])
            x_fit = np.array([x_values[mask].min(), x_values[mask].max()])
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, '-', lw=1.4, color=COLOR_PALETTE['fit_line'], alpha=0.8)
            ax.axhline(0, color='0.7', linestyle='--', linewidth=0.6)
            text_label = f'r = {r_value:.2f}'

        x_min, x_max = np.nanpercentile(x_values[mask], [2, 98])
        x_pad = (x_max - x_min) * 0.05 if x_max > x_min else max(abs(x_min) * 0.05, 1)
        y_min, y_max = np.nanpercentile(y_values[mask], [2, 98])
        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else max(abs(y_min) * 0.05, 1)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        if col == 'SW_diff': ax.set_xlim(-50, 50)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.text(
            0.05, 0.95,
            text_label,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=10
        )

    ax.grid(True, linestyle='--', alpha=0.2, color=COLOR_PALETTE['grid_color'])
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(width=1.05, labelsize=9)

plt.tight_layout()
figSelectedPath = current_dir + '/4_Figures/Rd1_R2_4_ET_residuals_selected_diagnostics_2x3'
plt.savefig(figSelectedPath, dpi=900)
# plt.close(fig_selected)

fig, axs = plt.subplots(1, 3, figsize=(13.2*0.8, 3.4*0.8))
axs = np.atleast_2d(axs)

for row, (group_label, et_col, tac_col) in enumerate(group_configs):
    df_group = df_tac.copy()
    df_group['urban_irri'] = df_group['all_urban_rural_et_residual']
    df_group['target_tac_diff'] = df_group[tac_col]

    ax_et = axs[row, 0]
    x_et = df_group['all_urban_rural_et_diff']
    y_et = df_group['all_urban_rural_et_residual']
    et_mask = ~np.isnan(x_et) & ~np.isnan(y_et)
    et_xlim = None
    et_ylim = None
    ax_et.plot(
        x_et[et_mask],
        y_et[et_mask],
        'o',
        alpha=0.5,
        color=COLOR_PALETTE['background_dots'],
        markersize=3
    )
    if np.sum(et_mask) > 2:
        slope, intercept, r_value, p_value, _ = st.linregress(x_et[et_mask], y_et[et_mask])
        x_fit = np.array([x_et[et_mask].min(), x_et[et_mask].max()])
        y_fit = slope * x_fit + intercept
        ax_et.plot(x_fit, y_fit, '-', lw=1.5, color=COLOR_PALETTE['fit_line'], alpha=0.8)
        x_min, x_max = np.nanpercentile(x_et[et_mask], [1, 99])
        x_pad = (x_max - x_min) * 0.05 if x_max > x_min else max(abs(x_min) * 0.05, 1)
        et_xlim = (x_min - x_pad, x_max + x_pad)
        y_min, y_max = np.nanpercentile(y_et[et_mask], [1, 99])
        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else max(abs(y_min) * 0.05, 1)
        et_ylim = (y_min - y_pad, y_max + y_pad)
        mean_residual = np.nanmean(y_et[et_mask])
        ax_et.text(
            0.05, 0.95,
            f'r = {r_value:.2f}\nmean residual = {mean_residual:.2f}',
            transform=ax_et.transAxes,
            ha='left',
            va='top',
            fontsize=10
        )

    if et_xlim is not None:
        ax_et.set_xlim(et_xlim)
    if et_ylim is not None:
        ax_et.set_ylim(et_ylim)
    ax_et.axhline(0, color='0.7', linestyle='--', linewidth=0.6)
    ax_et.axvline(0, color='0.7', linestyle='--', linewidth=0.6)
    ax_et.grid(True, linestyle='--', alpha=0.2, color=COLOR_PALETTE['grid_color'])
    ax_et.set_xlabel('Original ΔET (mm/year)', fontsize=11)
    ax_et.set_ylabel('Residual ΔET (mm/year)', fontsize=11)
    ax_et.set_title('Residual ΔET model', fontsize=12)
    ax_et.tick_params(width=1.05, labelsize=10)

    df_xgb = df_group[partial_corr_predictor_cols + ['target_tac_diff']].dropna()

    correlations, p_values = calculate_partial_correlations(df_xgb, 'target_tac_diff')
    print(correlations)

    ax_bar = axs[row, 1]
    bars = ax_bar.bar(
        range(len(correlations)),
        list(correlations.values()),
        color=COLOR_PALETTE['bar_color'],
        edgecolor=COLOR_PALETTE['scatter_edge'],
        alpha=0.8,
        hatch='//'
    )
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
            y_pos = height + 0.015 if height >= 0 else height - 0.06
            ax_bar.text(j, y_pos, marker, ha='center', fontsize=9)

    ax_bar.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax_bar.set_ylim([-0.42, 0.42])
    ax_bar.set_title(f'Partial r', fontsize=12)
    ax_bar.set_ylabel('Partial r', fontsize=11)
    ax_bar.set_xticks(range(len(correlations)))
    ax_bar.set_xticklabels(partial_corr_xtick_labels, rotation=45, ha='right', fontsize=10)
    ax_bar.tick_params(width=1.05, labelsize=10)

    ax_scatter = axs[row, 2]
    x_values = df_group['urban_irri']
    y_values = df_group['target_tac_diff']
    mask = ~np.isnan(x_values) & ~np.isnan(y_values)
    scatter_xlim = None
    scatter_ylim = None
    ax_scatter.plot(
        x_values[mask],
        y_values[mask],
        'o',
        alpha=0.5,
        color=COLOR_PALETTE['background_dots'],
        markersize=3
    )
    if np.sum(mask) > 2:
        slope, intercept, r_value, p_value, _ = st.linregress(x_values[mask], y_values[mask])
        x_fit = np.array([x_values[mask].min(), x_values[mask].max()])
        y_fit = slope * x_fit + intercept
        ax_scatter.plot(x_fit, y_fit, '-', lw=1.5, color=COLOR_PALETTE['fit_line'], alpha=0.8)
        ax_scatter.text(
            0.05, 0.95,
            f'r = {r_value:.2f}',
            transform=ax_scatter.transAxes,
            ha='left',
            va='top',
            fontsize=9
        )
        x_min, x_max = np.nanpercentile(x_values[mask], [0.5, 99.5])
        x_pad = (x_max - x_min) * 0.05 if x_max > x_min else max(abs(x_min) * 0.05, 1)
        scatter_xlim = (x_min - x_pad, x_max + x_pad)
        y_min, y_max = np.nanpercentile(y_values[mask], [0.5, 99.5])
        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else max(abs(y_min) * 0.05, 1)
        scatter_ylim = (y_min - y_pad, y_max + y_pad)
        print(group_label, r_value)

    ax_scatter.axhline(0, color='0.7', linestyle='--', linewidth=0.6)
    ax_scatter.axvline(0, color='0.7', linestyle='--', linewidth=0.6)
    ax_scatter.grid(True, linestyle='--', alpha=0.2, color=COLOR_PALETTE['grid_color'])
    if scatter_xlim is not None:
        ax_scatter.set_xlim(scatter_xlim)
    if scatter_ylim is not None:
        ax_scatter.set_ylim(scatter_ylim)

    ax_scatter.set_xlabel('Residual ΔET (mm/year)', fontsize=11)
    ax_scatter.set_ylabel('ΔTAC', fontsize=11)
    ax_scatter.set_title(f'ΔTAC vs Residual ΔET', fontsize=12)
    ax_scatter.tick_params(width=1.05, labelsize=10)

plt.tight_layout()
figToPath = current_dir + '/4_Figures/Rd1_R2_4_ET_residuals_all_vegetation'
plt.savefig(figToPath, dpi=900)
# plt.close(fig)
