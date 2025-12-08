## using validated method to calculate global urban irrigation
import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;

matplotlib.use('Qt5Agg')
import os
import rasterio
import scipy.stats as st
import seaborn as sns

def cal_ETa(Rn, Ts, Ta, Td):

    Rn = Rn*(1-0.23)
    ra = 165  # s/m
    Pa = 101 * 1000  # Pa
    rhoa = Pa / (287.05 * (Ta + 273.15))  # presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)
    ea = 2.1718e10 * np.exp(-4157. / (Td + 273.15 - 33.91));
    es = 2.1718e10 * np.exp(-4157. / (Ta + 273.15 - 33.91));
    rh = ea/es
    # specific heat of dry air
    Cp = 1005 + ((Ta + 273.15) - 250) ** 2 / 3364  # [J kg-1 K-1] (Garratt, 1994)
    # specific heat of air
    psy = rhoa * Cp / (ra * Rn)
    c = 1.
    ET_frc = 1-psy*(Ts*0.85-c*Ta).values*0.75#*rh.mean()
    ET_frc[ET_frc>1]=1
    ET_frc[ET_frc < 0] = 0
    ETa = (ETref-ETref.min()*0.5)*ET_frc*1000*0.75 # calculated ET; openET is reference
    # ETa[ETa>250]=np.nan
    return ETa

def add_fit_line(ax, x, y, color='black', lw=2, label='Fit'):
    x = np.array(x)
    y = np.array(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return
    slope, intercept, r, p, _ = st.linregress(x[mask], y[mask])
    xfit = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
    yfit = slope * xfit + intercept
    ax.plot(xfit, yfit, '-', color='k', lw=lw)
    # ax.legend(fontsize=10, loc='best', frameon=True)

df_et_dir = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'ET_era5L_csv_urban_751.csv')
df_et = pd.read_csv(df_et_dir)
et = df_et.iloc[:,1:-4].values

etref_df_dir = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'ETref_era5L_csv_urban_751.csv')
etref_df = pd.read_csv(etref_df_dir)

rn_path1 = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'Rn_lw_era5L_csv_urban_751.csv')
rn_df1 = pd.read_csv(rn_path1)

rn_path2 = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'Rn_sw_era5L_csv_urban_751.csv')
rn_df2 = pd.read_csv(rn_path2)

ta_path = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'Ta_era5L_csv_urban_751.csv')
ta_df = pd.read_csv(ta_path)

td_path = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'dewpoint_temperature_era5L_csv_urban_751.csv')
td_df = pd.read_csv(td_path)

lst_path = os.path.join('..', '2_Output', 'pure_veg', 'monthly_landsat_lst.csv')
lst_df = pd.read_csv(lst_path).copy()
our_ET_df = lst_df.copy()
IDs = np.sort(list(set(lst_df['id'])))

# 1. core
core_openET = []
core_ETa = []
for id in IDs:
    lst_i = lst_df[(lst_df['id'] == id)].copy()
    Ts = lst_i.iloc[:, 5:]
    if lst_i.empty: continue

    etref_i = etref_df[etref_df['ID'] == id].iloc[:, 1:-4].values * -1
    etref_reshaped = etref_i.reshape(etref_i.shape[0], 24, 12)
    ETref = np.nanmean(etref_reshaped, axis=1)
    ta_i = ta_df[ta_df['ID'] == id].iloc[:, 1:-4].values - 273.15
    ta_reshaped = ta_i.reshape(ta_i.shape[0], 24, 12)
    Ta = np.nanmean(ta_reshaped, axis=1)

    td_i = td_df.loc[td_df['ID'] == id]
    td = td_i.iloc[:, 1:-4].values - 273.15
    td_reshaped = td.reshape(td.shape[0], 24, 12)
    Td = np.nanmean(td_reshaped, axis=1)

    rn_df2_i = rn_df2[rn_df2['ID'] == id].iloc[:, 1:-4].values / (3600 * 24 * 30)
    rn_reshaped = rn_df2_i.reshape(rn_df2_i.shape[0], 24, 12)
    Rn = np.mean(rn_reshaped, axis=1)  # *1.3

    ETa = cal_ETa(Rn, Ts, Ta, Td)

    ETa_df = pd.DataFrame(ETa, index=lst_i.index)
    ETa_df.columns = ['et_m' + str(i + 1) for i in range(12)]
    our_ET_df.loc[lst_i.index, ['et_m1','et_m2','et_m3','et_m4','et_m5','et_m6','et_m7','et_m8','et_m9','et_m10','et_m11','et_m12']] = ETa_df.values
    print(id)

new_columns = our_ET_df.columns.tolist()
new_columns[5:17] = ['lst_m1','lst_m2','lst_m3','lst_m4','lst_m5','lst_m6',
                     'lst_m7','lst_m8','lst_m9','lst_m10','lst_m11','lst_m12']
our_ET_df.columns = new_columns
output_path = os.path.join('..', '2_Output', 'pure_veg', 'monthly_landsat_et_global.csv')
our_ET_df.to_csv(output_path, index=False)

