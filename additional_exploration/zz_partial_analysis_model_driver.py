# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")
import os
import geopandas as gpd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
import shap
import pingouin as pg

urban_factors = pd.read_csv(os.path.join('../..', '2_Output', 'drivers', 'urban_factors_effect.csv'))
co2 = pd.read_csv(os.path.join('../..', '2_Output', 'city_mean_CO2.csv'))
vpd = pd.read_csv(os.path.join('../..', '..', 'urban_env_data', 'vpd_csv_urban_751.csv'))
ntl_vegc_ndvi = pd.read_csv(os.path.join('../..', '2_Output', 'drivers', 'ntl_ndvi_veg.csv'))
pop = pd.read_csv(os.path.join('../..', '2_Output', 'drivers', 'pop.csv'))

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

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate['MAT'] = climate['MAT']-273.15
climate['MAP'] = climate['MAP']*24*1000

df_tac = pd.merge(df_tac, climate, on='ID')
df_tac = pd.merge(df_tac, urban_factors, on='ID')
df_tac = pd.merge(df_tac, co2, on='ID')
df_tac = pd.merge(df_tac, vpd, on='ID')
df_tac = pd.merge(df_tac, ntl_vegc_ndvi,on='ID')
df_tac = pd.merge(df_tac, pop,on='ID')

df_tac['ntl_diff'] = df_tac['ntl_core_all'] - df_tac['ntl_bg_all']
df_tac['pop_diff'] = df_tac['pop_core_all'] - df_tac['pop_bg_all']

df_tac['ndvi_diff'] = df_tac['ndvi_core_all'] - df_tac['ndvi_bg_all']
df_tac['tac_diff'] = df_tac['urban_core'] - df_tac['rural_bgr']
df_tac['co2_diff'] = df_tac['Urban_Core_CO2'] - df_tac['Urban_Outedge_CO2']+0.5
df_tac['LE_core'] = df_tac['Rn_core'] + df_tac['Q_core'] - df_tac['H_core'] - df_tac['Rn_core'] * 0.25
df_tac['LE_bg'] = df_tac['Rn_bg'] + df_tac['Q_bg'] - df_tac['H_bg'] - df_tac['Rn_bg'] * 0.25
df_tac['urban_irri'] = df_tac['LE_core']-df_tac['LE_bg']
df_tac['Q_diff'] = df_tac['Q_core'] - df_tac['Q_bg']

df_xgb = df_tac[['urban_irri','Q_diff', 'ntl_diff', 'pop_diff', 'uhi','co2_diff','MAT', 'MAP', 'SW', 'vpd', 'tac_diff']]
# climate: Ta, Pr, Rad, VPD
# urbanization factors: UHI, dCO2, urban irrigation, delta_vegC, delta_light, delta_PM2.5, vegetation type difference.
df_xgb = df_xgb.dropna()
st.linregress(df_xgb['MAP'],df_xgb['tac_diff'])

# Create correlation matrix
corr_matrix = df_xgb.corr()

# Create figure with specified size
plt.figure(figsize=(10, 8))

# Create heatmap
sns.heatmap(corr_matrix, 
            annot=True,          # Show correlation values
            cmap='RdBu_r',       # Red-Blue diverging colormap
            vmin=-1, vmax=1,     # Set correlation range
            center=0,            # Center colormap at 0
            fmt='.2f',           # Format correlation values to 2 decimal places
            square=True)         # Make cells square

# Customize plot
plt.title('Correlation Heatmap of Urban Environmental Factors', pad=20)
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join('../..', '4_Figures', 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()

# Prepare data
X = df_xgb.drop('tac_diff', axis=1)
y = df_xgb['tac_diff']

# Calculate partial correlations for each variable
partial_correlations = {}
for col in X.columns:
    # Get list of control variables (all variables except the current one)
    controlling_vars = [c for c in X.columns if c != col]
    
    # Calculate partial correlation using pingouin
    pcorr = pg.partial_corr(data=df_xgb, x=col, y='tac_diff', covar=controlling_vars)
    partial_correlations[col] = pcorr['r'].values[0]

# Create bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(partial_correlations)), 
               list(partial_correlations.values()), 
               color='lightblue', 
               edgecolor='black')

# Customize plot
plt.xticks(range(len(partial_correlations)), 
           list(partial_correlations.keys()), 
           rotation=45, 
           ha='right')
plt.ylabel('Partial Correlation Coefficient')
plt.title('Partial Correlation with TAC Difference')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., 
             height,
             f'{height:.2f}',
             ha='center', 
             va='bottom')

plt.tight_layout()
plt.savefig(os.path.join('../..', '4_Figures', 'partial_correlation_bars.png'), dpi=300, bbox_inches='tight')
plt.show()
