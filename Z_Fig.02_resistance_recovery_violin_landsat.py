import numpy as np
import matplotlib;
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
plt.rc('lines', linewidth=1.0)  # Increase default line width
plt.rc('axes', linewidth=1.0)   # Increase axes line width
plt.rc('grid', linewidth=1.0)   # Increase grid line width
plt.tick_params(width=1.0, labelsize=14)
plt.rc('xtick.major', size=4, width=1.0)    # Increase length and width of major ticks
plt.rc('ytick.major', size=4, width=1.0)    # Increase length and width of major ticks
plt.close()

import pandas as pd
import os

# Create time series
t = np.linspace(0, 140, 140)

# Function to create the disturbance and recovery pattern
def create_pattern(min_val, recovery_rate, noise_level=0.02):
    # Initial fluctuation
    y = np.random.normal(0, noise_level, len(t))
    
    # Disturbance and recovery
    disturbance_center = 40
    recovery_center = 40
    
    # Create disturbance
    disturbance = min_val * np.exp(-(t - disturbance_center)**2 / 50)
    disturbance[disturbance_center:] = 0
    
    # Create recovery
    recovery = np.where(t > recovery_center,
                       min_val * np.exp(-(t - recovery_center) * recovery_rate),
                       0)
    recovery[0:disturbance_center] = 0
    return y + disturbance + recovery

# Create three patterns
pattern1 = create_pattern(min_val=-0.5, recovery_rate=0.25)*0.1  # Quick recovery
# pattern2 = create_pattern(min_val=-1.0, recovery_rate=0.13)*0.1  # Medium recovery
pattern3 = create_pattern(min_val=-1.5, recovery_rate=0.11)*0.1  # Slow recovery

# Create figure
plt.figure(figsize=(11 * 0.72, 4 * 0.65))

# Plot patterns
plt.plot(t, pattern1, label='High resilience', color='#2166ac', linewidth=2)
# plt.plot(t, pattern2, label='Midum resilience', color='#67a9cf', linewidth=2)
plt.plot(t, pattern3, label='Low resilience', color='#b2182b', linewidth=2)

# Customize plot
plt.ylim(-0.2, 0.022)
plt.xlabel('Time (Weeks)')
plt.ylabel('VI anomaly')

# Adjust layout and save
plt.tight_layout()
plt.savefig('../4_Figures/Fig05_conceptual_recovery.png', dpi=900, bbox_inches='tight')
plt.show()


def remove_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[df_clean[col].between(lower_bound, upper_bound)]
    return df_clean

current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
urban_folder = current_dir + '/2_Output/nanFrac_urban_label_Landsat/'
folder = current_dir + '/2_Output/VI_Landsat/'
filenames = os.listdir(urban_folder)
names = []
for name in filenames:
    if name.startswith('label') and name.endswith('.npy'):
        names.append(name)
IDs = []
for name in names:
    id = name.split('_')[-1].split('.npy')[0]
    IDs.append(float(id))
IDs = np.sort(IDs)

recovery_stats = []  # List to store recovery statistics
IDs_num = []
sd_names = ['2.5']
for sd_name in sd_names:
    for id in IDs[1:]:

        dt_vi = np.load(current_dir + '/2_Output/Landsat_recovery_resistance/dVI_' + str(id) + '_smith_' + sd_name + 'sd.npy')
        urban_nonfrac_labels = np.load(urban_folder+'label_'+str(id)+'.npy')
        urban_lab = urban_nonfrac_labels[0,:]

        dt_vi_inner = np.nanmean(dt_vi[urban_lab == 2], axis=0)*-1
        dt_vi_sub = np.nanmean(dt_vi[urban_lab == 1], axis=0)*-1
        dt_vi_rural = np.nanmean(dt_vi[urban_lab == 0], axis=0)*-1

        recovery_stats.append([id,dt_vi_inner,dt_vi_sub,dt_vi_rural])
    recovery_stats_arr = np.array(recovery_stats)

    np.nanmean(recovery_stats_arr,axis=0)
    recovery_df = pd.DataFrame(recovery_stats_arr)
    recovery_df.columns = ['ID','dt_vi_inner','dt_vi_sub','dt_vi_rural']
    columns_to_check = ['dt_vi_inner','dt_vi_sub','dt_vi_rural']
    recovery_df = remove_outliers(recovery_df, columns_to_check)
    recovery_df = recovery_df.dropna()

# load Landsat TAC
relative_path = '/2_Output/tac_landsat_city_3zones_exclude_disturbance'+sd_name+'.npz'
TACs = np.load(current_dir+relative_path)['array1'] # v5,v4.2
TACs_mean = np.nanmean(TACs, axis=2)
ID = np.load(current_dir+relative_path)['array2'] # v5,v4.2
df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_edge','rural_bgr']
df_tac['ID'] = ID

df_tac = pd.merge(recovery_df,df_tac,on='ID')
# Create figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11 * 0.72, 4 * 0.65))

df_tac['urban_core'][df_tac['urban_core']>np.nanpercentile(df_tac['urban_core'],95)]=np.nan
df_tac['rural_bgr'][df_tac['rural_bgr']>np.nanpercentile(df_tac['rural_bgr'],95)]=np.nan

# Data for plots
resistance_data = [df_tac['dt_vi_inner'], df_tac['dt_vi_rural']]
taced_data = [df_tac['urban_core'].dropna(), df_tac['rural_bgr'].dropna()]
colors = ['#2166ac', '#b2182b']
labels = ['UC', 'RA']

# Plot resistance
parts1 = ax1.violinplot(resistance_data, showmeans=False, showmedians=False, showextrema=False)
for pc, color in zip(parts1['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

ax1.boxplot(resistance_data, labels=labels, showfliers=False, patch_artist=True,
            boxprops=dict(facecolor='none', edgecolor='black'),
            medianprops=dict(color='black'))
ax1.set_ylabel('Resistance (ΔkNDVI)')
ax1.set_ylim(resistance_data[0].min()-0.002, resistance_data[-1].max() * 1.15)
# Add labels on top of each violin bar
ax1.text(1.5, ax1.get_ylim()[1] * 0.95, '***', ha='center', va='top', fontsize=10, fontweight='bold')
# ax1.text(2, ax1.get_ylim()[1] * 0.95, 'b', ha='center', va='top', fontsize=10, fontweight='bold')

# Plot recovery
parts2 = ax2.violinplot(taced_data, showmeans=False, showmedians=False, showextrema=False)
for pc, color in zip(parts2['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

ax2.boxplot(taced_data, labels=labels, showfliers=False, patch_artist=True,
            boxprops=dict(facecolor='none', edgecolor='black'),
            medianprops=dict(color='black'))
ax2.set_ylabel('TAC${_ED}$')
ax2.set_ylim(taced_data[0].min()-0.01, taced_data[-1].max()*1.15)
# Add labels on top of each violin bar
ax2.text(1.5, ax2.get_ylim()[1] * 0.95, '***', ha='center', va='top', fontsize=10, fontweight='bold')
# ax2.text(2, ax2.get_ylim()[1] * 0.95, 'b', ha='center', va='top', fontsize=10, fontweight='bold')

df_merge = df_tac
tac_uc = df_merge['urban_core']
tac_rb = df_merge['rural_bgr']

dt_vi_uc = df_merge['dt_vi_inner']
dt_vi_rb = df_merge['dt_vi_rural']
plt.figure(); plt.plot(tac_uc-tac_rb,dt_vi_uc-dt_vi_rb,'o')
import scipy.stats as st
mask = np.isnan(tac_uc-tac_rb-dt_vi_uc-dt_vi_rb)
st.linregress(tac_uc[~mask]-tac_rb[~mask],dt_vi_uc[~mask]-dt_vi_rb[~mask])

sum(tac_uc-tac_rb>0)

# x_fit10 = np.array([dt_vi_uc.min(), dt_vi_uc.max()])
x_fit10 = x_fit12 = np.array([dt_vi_rb.min(), dt_vi_rb.max()])

import scipy.stats as st
# Second subplot: TAC vs Recovery
slope20, intercept20, r_value20, p_value20, std_err20 = st.linregress( dt_vi_uc[~np.isnan(tac_uc)],tac_uc[~np.isnan(tac_uc)])
slope22, intercept22, r_value22, p_value22, std_err22 = st.linregress( dt_vi_rb[~np.isnan(tac_rb)],tac_rb[~np.isnan(tac_rb)])

ax3.scatter(dt_vi_uc, tac_uc, color=colors[0], s=18, alpha=0.1)
ax3.scatter(dt_vi_rb, tac_rb, color=colors[1], s=18, alpha=0.1)

ax3.plot(x_fit10, slope20 * x_fit10 + intercept20, colors[0], linewidth=1.5)
ax3.plot(x_fit12, slope22 * x_fit12 + intercept22, colors[1], linewidth=1.5)

ax3.set_ylabel('TAC$_{ED}$')
ax3.set_xlabel('ΔkNDVI')


plt.tight_layout()
plt.savefig(os.path.join(current_dir, '4_Figures', 'recovery_resistance_TAC.png'), dpi=900)
plt.show()

# # Perform statistical test
import scipy.stats as st
print("T-test results (inner vs rural recovery):")

print(st.ttest_rel(dt_vi_uc, dt_vi_rb))
print(st.ttest_rel(tac_uc[~mask], tac_rb[~mask]))
