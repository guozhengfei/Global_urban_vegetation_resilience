# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
plt.close()
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

AI = pd.read_csv(current_dir + '/2_Output/AI_csv.csv')
ch = pd.read_csv(current_dir + '/2_Output/drivers/canopy_height.csv')

df_tac = pd.merge(df_tac, climate, on='ID')
df_tac = pd.merge(df_tac, urban_factors, on='ID')
df_tac = pd.merge(df_tac, co2, on='ID')
df_tac = pd.merge(df_tac, vpd, on='ID')
df_tac = pd.merge(df_tac, ntl_vegc_ndvi, on='ID')
df_tac = pd.merge(df_tac, pop, on='ID')
df_tac = pd.merge(df_tac, AI, on='ID')
df_tac = pd.merge(df_tac, ch, on='ID')

df_tac['ntl_diff'] = df_tac['ntl_core_all'] - df_tac['ntl_bg_all']
df_tac['pop_diff'] = df_tac['pop_core_all'] - df_tac['pop_bg_all']

df_tac['fvc_diff'] = df_tac['veg_core_all'] - df_tac['veg_bg_all']
df_tac['tac_diff'] = df_tac['urban_core'] - df_tac['rural_bgr']
df_tac['co2_diff'] = df_tac['Urban_Core_CO2'] - df_tac['Urban_Outedge_CO2']+0.5
df_tac['LE_core'] = df_tac['Rn_core'] + df_tac['Q_core'] - df_tac['H_core'] - df_tac['Rn_core'] * 0.25
df_tac['LE_bg'] = df_tac['Rn_bg'] + df_tac['Q_bg'] - df_tac['H_bg'] - df_tac['Rn_bg'] * 0.25
df_tac['urban_irri'] = df_tac['LE_core']-df_tac['LE_bg']
df_tac['Q_diff'] = df_tac['Q_core'] - df_tac['Q_bg']
df_tac['ch_diff'] = df_tac['ch_core'] - df_tac['ch_bg']

df_xgb = df_tac[['SW', 'AI', 'urban_irri', 'ntl_diff', 'pop_diff', 'uhi', 'co2_diff',  'ch_diff', 'tac_diff']]#'vpd',
# climate: Ta, Pr, Rad, VPD
# urbanization factors: UHI, dCO2, urban irrigation, delta_vegC, delta_light, delta_PM2.5, vegetation type difference.
df_xgb = df_xgb.dropna()

# Create correlation matrix
corr_matrix = df_xgb.corr()

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
plt.savefig(os.path.join('../..', '4_Figures', 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()

# Prepare data
X = df_xgb.drop('tac_diff', axis=1)
y = df_xgb['tac_diff']

# # Initialize lists to store performance metrics
# y_tests = []
# y_preds = []

# # Initialize leave-one-out cross-validation
# loo = LeaveOneOut()

# # Train and evaluate model using leave-one-out cross-validation
# for train_index, test_index in loo.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#     # Initialize and train XGBoost model
#     model = xgb.XGBRegressor(n_estimators=100,
#     learning_rate=0.1,
#     max_depth=3,
#     random_state=50)
#     model.fit(X_train, y_train)

#     # Make predictions
#     y_pred = model.predict(X_test)

#     # Calculate and store performance metrics
#     y_tests.extend(y_test)
#     y_preds.extend(y_pred)


# print(f"{st.linregress(y_preds,y_tests)}")
# # Create scatter plot of predicted vs actual values
# plt.figure(figsize=(8, 8))
# plt.scatter(y_tests, y_preds, alpha=0.5)
# plt.plot([min(y_tests), max(y_tests)], [min(y_tests), max(y_tests)], 'r--')
# plt.xlabel('Actual TAC difference')
# plt.ylabel('Predicted TAC difference')

plt.tight_layout()
plt.savefig(os.path.join('../..', '4_Figures', 'predicted_vs_actual.png'),
            dpi=900, bbox_inches='tight')
plt.show()

# Train final model on all data for SHAP analysis
final_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
final_model.fit(X, y)

# Calculate SHAP values
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

# Define factor categories
urban_factors = ['urban_irri', 'ntl_diff', 'pop_diff', 'uhi', 'co2_diff']
background_factors = ['AI', 'SW'] # vpd

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.8, 5),
                              gridspec_kw={'height_ratios': [2, 1]})# 10*0.52, 8*0.5

# Calculate mean absolute SHAP values for each feature
mean_shap_values = np.abs(shap_values).mean(0)
feature_importance = pd.Series(mean_shap_values, index=X.columns)

# Create color map for bars
colors = ['#FF9999' if feat in urban_factors else '#66B2FF' for feat in X.columns]

# Sort features by importance
feature_importance_sorted = feature_importance.sort_values(ascending=True)
colors_sorted = [colors[list(X.columns).index(feat)] for feat in feature_importance_sorted.index]

# Create bar plot
bars = ax1.barh(range(len(feature_importance_sorted)), 
                feature_importance_sorted,
                color=colors_sorted)

# Customize bar plot
ax1.set_yticks(range(len(feature_importance_sorted)))
ax1.set_yticklabels(feature_importance_sorted.index)
ax1.set_xlabel('mean|SHAP value|')
# ax1.set_title('Feature Importance')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#FF9999', label='Urbanization Factors'),
                  Patch(facecolor='#66B2FF', label='Background Factors')]
ax1.legend(handles=legend_elements, loc='lower right')

# Calculate total importance for each category
urban_importance = feature_importance[urban_factors].sum()
background_importance = feature_importance[background_factors].sum()
total_importance = urban_importance + background_importance

# Create pie chart
sizes = [urban_importance, background_importance]
labels = ['Urbanization\nFactors', 'Background\nFactors']
colors_pie = ['#FF9999', '#66B2FF']
ax2.pie(sizes, labels=labels, colors=colors_pie,
        autopct='%1.1f%%', startangle=90)

plt.tight_layout()
plt.savefig(os.path.join('../..', '4_Figures', 'shap_importance_landsat.png'),
            dpi=900, bbox_inches='tight')
plt.show()

# Get top 4 important features
top_4_features = feature_importance_sorted.index[-4:]

# Create scatter plots for top 4 features
plt.figure(figsize=(6.2, 4))

for i in range(4):
    plt.subplot(2, 2, i+1)
    
    # Determine variables
    val_label = ['urban_irri', 'pop_diff', 'AI', 'AI'][::-1]
    col_label = ['AI', 'AI', 'urban_irri', 'uhi'][::-1]
    
    # Get current feature values and SHAP values
    x_values = X[val_label[i]]
    y_values = shap_values[:, list(X.columns).index(val_label[i])]
    
    # Create scatter plot with gradient color
    scatter = plt.scatter(x_values,
                         y_values,
                         c=X[col_label[i]],
                         cmap='BrBG',
                         s=5,
                         alpha=0.6,
                         vmax=np.nanpercentile(X[col_label[i]], 90),
                         vmin=np.nanpercentile(X[col_label[i]], 5))
    
    # Calculate binned means for smooth line
    bins = np.linspace(x_values.min(), x_values.max(), 21)  # 20 bins + 1 edge
    digitized = np.digitize(x_values, bins)
    bin_means = [y_values[digitized == j].mean() for j in range(1, len(bins))]
    bin_means[-1] = y_values[x_values==x_values.max()][0]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Fill NaN values with interpolation
    bin_means = pd.Series(bin_means).interpolate(method='linear').values
    
    # Apply moving average while keeping edge values
    smoothed_means = np.copy(bin_means)
    for i0 in range(2, len(bin_means) - 2):  # Skip first two and last two points
        smoothed_means[i0] = np.mean(bin_means[i0-2:i0+3])
    
    # Plot smooth line
    plt.plot(bin_centers, smoothed_means, 'k-', linewidth=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    
    # Customize plot
    plt.xlabel(val_label[i])
    plt.ylabel(f'SHAP value for {val_label[i]}')

plt.tight_layout()
plt.savefig(os.path.join('../..', '4_Figures', 'shap_scatter_landsat.png'),
            dpi=900, bbox_inches='tight')
plt.show()