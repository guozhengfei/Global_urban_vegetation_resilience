## 1. using openET to test  our PM ET which is estimated using LST
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

def cal_ETa_PM(Rn, Ts, Ta, Td):

    Rn = Rn#*(1-0.23)
    ra = 195  # s/m
    Pa = 101 * 1000  # Pa
    rhoa = Pa / (287.05 * (Ta + 273.15))  # presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)
    ea = 2.1718e10 * np.exp(-4157. / (Td + 273.15 - 33.91));
    es = 2.1718e10 * np.exp(-4157. / (Ta + 273.15 - 33.91));
    rh = ea/es
    # specific heat of dry air
    Cp = 1005 + ((Ta + 273.15) - 250) ** 2 / 3364  # [J kg-1 K-1] (Garratt, 1994)

    H = rhoa * Cp * (Ts - Ta) / ra
    H[H<0]=0
    G = Rn*0.15
    LE = Rn - H - G
    LE[LE<0]=np.nan
    lmt = 24*3600*30/(2.45*10**6)
    ET = LE/lmt
    # ensure numpy array
    ET = np.array(ET, dtype=float)
    # remove baseline per-row
    mins = np.nanmin(ET, axis=1)
    ET_final = ET - mins[:, None]
    return ET/0.72


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

    # Add text box with slope and R value
    stats_text = f'Slope = {slope:.2f}\nR = {r:.2f}'
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
            fontsize=9, verticalalignment='top')

    return slope, r

df_et_dir = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'ET_era5L_csv_urban_751.csv')
df_et = pd.read_csv(df_et_dir)
et = df_et.iloc[:,1:-4].values

etref_df_dir = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'ETref_era5L_csv_urban_751.csv')
etref_df = pd.read_csv(etref_df_dir)

openet_path = os.path.join('..', '2_Output', 'pure_veg', 'monthly_openET_2001_2023.csv')
openet_df = pd.read_csv(openet_path)

rn_path1 = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'Rn_lw_era5L_csv_urban_751.csv')
rn_df1 = pd.read_csv(rn_path1)

rn_path2 = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'Rn_sw_era5L_csv_urban_751.csv') # net solar radiation
rn_df2 = pd.read_csv(rn_path2)

ta_path = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'Ta_era5L_csv_urban_751.csv')
ta_df = pd.read_csv(ta_path)

td_path = os.path.join('..','..', 'urban_env_data', 'ERA5_Land', 'dewpoint_temperature_era5L_csv_urban_751.csv')
td_df = pd.read_csv(td_path)

lst_path = os.path.join('..', '2_Output', 'pure_veg', 'monthly_landsat_lst.csv')
lst_df = pd.read_csv(lst_path)
IDs = np.sort(list(set(openet_df['id'])))

# keep lst_df and openet_df rows with same lat and lon
common_ids = set(lst_df['id']).intersection(set(openet_df['id']))
common_lats = set(lst_df['lat']).intersection(set(openet_df['lat']))
common_lon = set(lst_df['lon']).intersection(set(openet_df['lon']))

lst_df = lst_df[lst_df['lon'].isin(common_lon) & lst_df['lat'].isin(common_lats)]
openet_df = openet_df[openet_df['lon'].isin(common_lon) & openet_df['lat'].isin(common_lats)]

# Update IDs after filtering
IDs = np.sort(list(set(openet_df['id'])))
fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.7))

# 1. core
# plt.figure()
core_openET = []
core_ETa = []
for id in IDs:
    openet_i = openet_df[(openet_df['id'] == id) & (openet_df['type'] == 'core')].copy()
    openet = openet_i.iloc[:, 5:].values
    openet_reshaped = openet.reshape(openet.shape[0], 23, 12)
    openET = np.nanmean(openet_reshaped, axis=1)

    lst_i = lst_df[(lst_df['id'] == id) & (lst_df['type'] == 'core')].copy()
    Ts = lst_i.iloc[:, 5:]

    if openet_i.empty or lst_i.empty:
        continue
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

    rn_df2_i = rn_df2[rn_df2['ID'] == id].iloc[:, 1:-4].values / (3600 * 24 * 30) # net solar radiation
    rn_reshaped = rn_df2_i.reshape(rn_df2_i.shape[0], 24, 12)

    Rn = np.mean(rn_reshaped, axis=1)  # *1.3

    ETa = cal_ETa_PM(Rn, Ts, Ta, Td)
    for i in range(ETa.shape[0]):
        # slope =st.linregress(openET[i,:],ETa[i,:]).slope
        #
        # if (slope-1>1) | (np.nanmax(ETa[i,:])>275): continue
        core_openET.extend(openET.flatten().tolist())
        core_ETa.extend(ETa.flatten().tolist())
        # plt.plot(openET[i,:],ETa[i,:], 'o')

axs[0].scatter(core_openET, core_ETa, s=1, alpha=0.4, c='tab:blue')
axs[0].set_xlim([-50,300])
axs[0].set_ylim([-50,300])
axs[0].set_xlabel('OpenET (mm/month)')
axs[0].set_ylabel('Estimated ET (mm/month)')

# 2. bg
bg_openET = []
bg_ETa = []
for id in IDs:
    openet_i = openet_df[(openet_df['id'] == id) & (openet_df['type'] == 'bg')]
    openet = openet_i.iloc[:, 5:].values
    openet_reshaped = openet.reshape(openet.shape[0], 23, 12)
    openET = np.nanmean(openet_reshaped, axis=1)

    lst_i = lst_df[(lst_df['id'] == id) & (lst_df['type'] == 'bg')]
    Ts = lst_i.iloc[:, 5:]

    if openet_i.empty or lst_i.empty:
        continue

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
    Rn = np.mean(rn_reshaped, axis=1)

    ETa = cal_ETa_PM(Rn,Ts,Ta,Td)
    for i in range(ETa.shape[0]):
        # r = st.linregress(openET[i,:],ETa[i,:]).rvalue
        # slope =st.linregress(openET[i,:],ETa[i,:]).slope
        # if (slope-1>1) | (np.nanmax(ETa[i,:])>275): continue
        bg_openET.extend(openET.flatten().tolist())
        bg_ETa.extend(ETa.flatten().tolist())

axs[1].scatter(bg_openET, bg_ETa, s=1, alpha=0.3, c='tab:orange')
# sns.kdeplot(
#     x=bg_openET, y=bg_ETa,
#     fill=True, cmap="RdBu_r", thresh=0.06, levels=100,alpha=0.4,
#     ax=axs[0,1]
# )
axs[1].set_xlim([-50,300])
axs[1].set_ylim([-50,300])
axs[1].set_xlabel('OpenET (mm/month)')
axs[1].set_ylabel('Estimated ET (mm/month)')

# 3. core + bg
axs[2].scatter(core_openET, core_ETa, s=1, alpha=0.3, c='tab:blue', label='Urban')
axs[2].scatter(bg_openET, bg_ETa, s=1, alpha=0.3, c='tab:orange', label='Rural')

axs[2].set_xlabel('OpenET (mm/month)')
axs[2].set_ylabel('Estimated ET (mm/month)')
axs[2].legend()
openETs = np.array(core_openET+bg_openET)
ETas = np.array(core_ETa+bg_ETa)
mask = np.isnan(openETs)| np.isnan(ETas)
print(st.linregress(openETs[~mask],ETas[~mask]))
# add fit line for [1,0]
all_openET = np.array(core_openET + bg_openET)
all_ETa = np.array(core_ETa + bg_ETa)
add_fit_line(axs[2], all_openET, all_ETa, color='black', label='Fit')

# sns.kdeplot(
#     x=openETs, y=ETas,
#     fill=True, cmap="RdBu_r", thresh=0.06, levels=100,alpha=0.4,
#     ax=axs[1,0]
# )
axs[2].set_xlim([-50,300])
axs[2].set_ylim([-50,300])

core_openET =np.array(core_openET)
core_ETa =np.array(core_ETa)
mask = np.isnan(core_openET)| np.isnan(core_ETa)
print(st.linregress(core_openET[~mask],core_ETa[~mask]))
# add fit line for [0,0]

bg_openET =np.array(bg_openET)
bg_ETa =np.array(bg_ETa)
mask = np.isnan(bg_openET)| np.isnan(bg_ETa)
print(st.linregress(bg_openET[~mask],bg_ETa[~mask]))
# add fit line for [0,1]

# 4. core mean - bg mean for each city
core_bg_diff_openET = []
core_bg_diff_ETa = []
i=0
for id in IDs:
    openet_core = openet_df[(openet_df['id'] == id) & (openet_df['type'] == 'core')].iloc[:, 5:].values
    openet_reshaped = openet_core.reshape(openet_core.shape[0], 23, 12)
    openET_core = np.nanmean(openet_reshaped, axis=1)

    openet_bg = openet_df[(openet_df['id'] == id) & (openet_df['type'] == 'bg')].iloc[:, 5:].values
    openet_reshaped_bg = openet_bg.reshape(openet_bg.shape[0], 23, 12)
    openET_bg = np.nanmean(openet_reshaped_bg, axis=1)

    lst_core = lst_df[(lst_df['id'] == id) & (lst_df['type'] == 'core')].iloc[:, 5:]
    lst_bg = lst_df[(lst_df['id'] == id) & (lst_df['type'] == 'bg')].iloc[:, 5:]

    if len(openet_core) == 0 or len(openet_bg) == 0 or lst_core.empty or lst_bg.empty:
        continue
    openet_diff = np.nanmean(openET_core,axis=0) - np.nanmean(openET_bg,axis=0)

    # 计算ETa差值
    # core
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
    Rn = np.mean(rn_reshaped, axis=1)

    Ts_core = lst_core
    Ts_bg = lst_bg

    ETa_core = cal_ETa_PM(Rn,Ts_core,Ta, Td)
    ETa_core_mean = np.nanmean(ETa_core,axis=0)
    ETa_bg = cal_ETa_PM(Rn, Ts_bg, Ta, Td)
    ETa_bg_mean = np.nanmean(ETa_bg, axis=0)

    ETa_diff = ETa_core_mean - ETa_bg_mean
    core_bg_diff_openET.extend(openet_diff.tolist())
    core_bg_diff_ETa.extend(ETa_diff.tolist())
    i=i+1


axs[3].scatter(core_bg_diff_openET, core_bg_diff_ETa, s=15, fc='none',alpha=0.7, c='grey')
core_bg_diff_openET = np.array(core_bg_diff_openET)
core_bg_diff_openET[core_bg_diff_openET>100]=np.nan
core_bg_diff_ETa = np.array(core_bg_diff_ETa)
mask = np.isnan(core_bg_diff_openET) | np.isnan(core_bg_diff_ETa)
print(st.linregress(core_bg_diff_openET[~mask], core_bg_diff_ETa[~mask]),i)
axs[3].set_xlabel('OpenET-based UI (mm/month)')
axs[3].set_ylabel('Our ET-based UI (mm/month)')
axs[3].set_xlim([-55,95])
axs[3].set_ylim([-55,95])

# 1. core
slope1, r1 = add_fit_line(axs[0], core_openET, core_ETa, color='tab:blue', label='Fit')

# 2. bg
slope2, r2 = add_fit_line(axs[1], bg_openET, bg_ETa, color='tab:orange', label='Fit')

# 3. combined plot (already has legend, keep existing fit line)
slope3, r3 = add_fit_line(axs[2], all_openET, all_ETa, color='black', label='Fit')

# 4. core mean - bg mean for each city
slope4, r4 = add_fit_line(axs[3], core_bg_diff_openET, core_bg_diff_ETa, color='grey', label='Fit')

plt.tight_layout(w_pad=0.04)
outpath = os.path.join('..','4_Figures', 'openet_vs_eta_PM_4panels_with_fit.png')
plt.savefig(outpath, dpi=600)
plt.show()