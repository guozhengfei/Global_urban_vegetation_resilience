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
TACs = np.load(current_dir+'/2_Output/tac_landsat_city_3zones.npz')['array1']  # v5,v4.2
ID = np.load(current_dir+'/2_Output/tac_landsat_city_3zones.npz')['array2']  # v5,v4.2
co2 = pd.read_csv(os.path.join('..', '2_Output', 'city_mean_CO2.csv'))

shann = pd.read_csv(os.path.join('..', '2_Output', 'drivers','shann.csv'))

TACs_mean = np.nanmean(TACs, axis=2)
tac_global_mean = np.nanmean(TACs_mean, axis=0)

df_tac = pd.DataFrame(TACs_mean)
df_tac.columns = ['urban_core', 'urban_edge', 'rural_bgr']
df_tac['ID'] = ID

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate['MAT'] = climate['MAT'] - 273.15
climate['MAP'] = climate['MAP'] * 24 * 1000

urban_factor = pd.read_csv(current_dir + '/2_Output/drivers/urban_factors_effect.csv')
vpd = pd.read_csv(os.path.join('..', '..', 'urban_env_data', 'vpd_csv_urban_751.csv'))

df_tac = pd.merge(df_tac, climate, on='ID')
df_tac = pd.merge(df_tac, urban_factor, on='ID')
df_tac = pd.merge(df_tac, co2, on='ID')
df_tac = pd.merge(df_tac, vpd, on='ID')
df_tac = pd.merge(df_tac, shann, on='ID')
df_tac['dT'] = df_tac['Tc_core'] - df_tac['Tc_bg']
df_tac['dT'][(df_tac['dT'] > 10) | (df_tac['dT'] < -10)] = np.nan
df_tac = df_tac.dropna()

LE_core = df_tac['Rn_core'] + df_tac['Q_core'] - df_tac['H_core'] - df_tac['Rn_core'] * 0.3
LE_bg = df_tac['Rn_bg'] + df_tac['Q_bg'] - df_tac['H_bg'] - df_tac['Rn_bg'] * 0.3

fig, ax = plt.subplots(2, 2, figsize=(10*0.52, 8*0.5))

# Subplot 1: LE_core - LE_bg vs urban_core - rural_bgr
x1 = LE_core - LE_bg
y1 = df_tac['urban_core'] - df_tac['rural_bgr']
ax[0, 0].plot(x1, y1, 'o', mfc='none', color='C0')
slope1, intercept1, r1, _, _ = st.linregress(x1, y1)
ax[0, 0].plot(x1, slope1 * x1 + intercept1, 'r-', label=f'r={r1:.2f}')
ax[0, 0].set_xlabel('urban irrigation')
ax[0, 0].set_ylabel('ΔTAC')
ax[0, 0].legend()

# Subplot 2: uhi vs urban_core - rural_bgr
x2 = df_tac['uhi']
y2 = df_tac['urban_core'] - df_tac['rural_bgr']
ax[0, 1].plot(x2, y2, 'o', mfc='none', color='C0')
slope2, intercept2, r2, _, _ = st.linregress(x2, y2)
ax[0, 1].plot(x2, slope2 * x2 + intercept2, 'r-', label=f'r={r2:.2f}')
ax[0, 1].set_xlabel('UHI')
ax[0, 1].set_ylabel('ΔTAC')
ax[0, 1].legend()

# Subplot 3: Urban_Core_CO2 - Urban_Outedge_CO2 vs urban_core - rural_bgr
x3 = df_tac['Urban_Core_CO2'] - df_tac['Urban_Outedge_CO2']
y3 = df_tac['urban_core'] - df_tac['rural_bgr']
ax[1, 0].plot(x3, y3, 'o', mfc='none', color='C0')
slope3, intercept3, r3, _, _ = st.linregress(x3, y3)
ax[1, 0].plot(x3, slope3 * x3 + intercept3, 'r-', label=f'r={r3:.2f}')
ax[1, 0].set_xlabel('CO2')
ax[1, 0].set_ylabel('ΔTAC')
ax[1, 0].legend()

# Subplot 4: vpd vs urban_core - rural_bgr
x4 = df_tac['vpd']
y4 = df_tac['urban_core'] - df_tac['rural_bgr']
ax[1, 1].plot(x4, y4, 'o', mfc='none', color='C0')
slope4, intercept4, r4, _, _ = st.linregress(x4, y4)
ax[1, 1].plot(x4, slope4 * x4 + intercept4, 'r-', label=f'r={r4:.2f}')
ax[1, 1].set_xlabel('VPD')
ax[1, 1].set_ylabel('ΔTAC')
ax[1, 1].legend()

fig.tight_layout()
plt.savefig(current_dir + '/4_Figures/Fig04_TAC_diff_drivers.png', dpi=900)
plt.show()
