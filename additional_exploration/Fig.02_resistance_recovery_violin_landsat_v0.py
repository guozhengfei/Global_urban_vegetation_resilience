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
pattern1 = create_pattern(min_val=-0.5, recovery_rate=0.15)*0.1  # Quick recovery
pattern2 = create_pattern(min_val=-1.0, recovery_rate=0.13)*0.1  # Medium recovery
pattern3 = create_pattern(min_val=-1.5, recovery_rate=0.11)*0.1  # Slow recovery

# Create figure
plt.figure(figsize=(11 * 0.72, 3.5*0.7))

# Plot patterns
plt.plot(t, pattern1, label='High resilience', color='#2166ac', linewidth=2)
plt.plot(t, pattern2, label='Midum resilience', color='#67a9cf', linewidth=2)
plt.plot(t, pattern3, label='Low resilience', color='#b2182b', linewidth=2)

# Customize plot
plt.ylim(-0.2, 0.022)
plt.xlabel('Time (Days)')
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
sd_names = ['2']
for sd_name in sd_names:
    for id in IDs[1:]:
        resis = np.load(current_dir+'/2_Output/Landsat_recovery_resistance/resistance_'+str(id)+ '_smith_'+sd_name+'sd.npy')
        resil = np.load(current_dir+'/2_Output/Landsat_recovery_resistance/resilience_'+str(id)+ '_smith_'+sd_name+'sd.npy')
        dt_vi = np.load(current_dir + '/2_Output/Landsat_recovery_resistance/dVI_' + str(id) + '_smith_' + sd_name + 'sd.npy')
        urban_nonfrac_labels = np.load(urban_folder+'label_'+str(id)+'.npy')
        urban_lab = urban_nonfrac_labels[0,:]

        resis_inner = np.nanmean(resis[urban_lab == 2], axis=0)
        resis_sub = np.nanmean(resis[urban_lab == 1], axis=0)
        resis_rural = np.nanmean(resis[urban_lab == 0], axis=0)
        resil_inner = np.nanmean(resil[urban_lab == 2], axis=0)
        resil_sub = np.nanmean(resil[urban_lab == 1], axis=0)
        resil_rural = np.nanmean(resil[urban_lab == 0], axis=0)
        dt_vi_inner = np.nanmean(dt_vi[urban_lab == 2], axis=0)
        dt_vi_sub = np.nanmean(dt_vi[urban_lab == 1], axis=0)
        dt_vi_rural = np.nanmean(dt_vi[urban_lab == 0], axis=0)

        recovery_stats.append([id,resis_inner,resis_sub,resis_rural,resil_inner,resil_sub,resil_rural,dt_vi_inner,dt_vi_sub,dt_vi_rural])
    recovery_stats_arr = np.array(recovery_stats)

    np.nanmean(recovery_stats_arr,axis=0)
    recovery_df = pd.DataFrame(recovery_stats_arr)
    recovery_df.columns = ['ID','inner_rt','sub_rt','rural_rt','inner_recovery','sub_recovery','rural_recovery','dt_vi_inner','dt_vi_sub','dt_vi_rural']
    columns_to_check = ['inner_rt','sub_rt','rural_rt','inner_recovery','sub_recovery','rural_recovery','dt_vi_inner','dt_vi_sub','dt_vi_rural']
    recovery_df = remove_outliers(recovery_df, columns_to_check)
    recovery_df = recovery_df.dropna()

# Create figure with two subplots
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(11 * 0.72, 6 * 0.65))

# Data for plots
resistance_data = [recovery_df['inner_rt'], recovery_df['sub_rt'], recovery_df['rural_rt']]
recovery_data = [recovery_df['inner_recovery'], recovery_df['sub_recovery'], recovery_df['rural_recovery']]
colors = ['#2166ac', '#67a9cf', '#b2182b']
labels = ['UC', 'UE', 'RA']

# Plot resistance
parts1 = ax1.violinplot(resistance_data, showmeans=False, showmedians=False, showextrema=False)
for pc, color in zip(parts1['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

ax1.boxplot(resistance_data, labels=labels, showfliers=False, patch_artist=True,
            boxprops=dict(facecolor='none', edgecolor='black'),
            medianprops=dict(color='black'))
ax1.set_ylabel('Resistance')
ax1.set_ylim(recovery_df['rural_rt'].min() * 1.1, 0)
# Add labels on top of each violin bar
ax1.text(1, ax1.get_ylim()[1] * 0.95, 'a', ha='center', va='top', fontsize=10, fontweight='bold')
ax1.text(2, ax1.get_ylim()[1] * 0.95, 'b', ha='center', va='top', fontsize=10, fontweight='bold')
ax1.text(3, ax1.get_ylim()[1] * 0.95, 'c', ha='center', va='top', fontsize=10, fontweight='bold')


# Plot recovery
parts2 = ax2.violinplot(recovery_data, showmeans=False, showmedians=False, showextrema=False)
for pc, color in zip(parts2['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

ax2.boxplot(recovery_data, labels=labels, showfliers=False, patch_artist=True,
            boxprops=dict(facecolor='none', edgecolor='black'),
            medianprops=dict(color='black'))
ax2.set_ylabel('Recovery')
ax2.set_ylim(0, recovery_df['rural_recovery'].max() * 1.2)
# Add labels on top of each violin bar
ax2.text(1, ax2.get_ylim()[1] * 0.95, 'a', ha='center', va='top', fontsize=10, fontweight='bold')
ax2.text(2, ax2.get_ylim()[1] * 0.95, 'b', ha='center', va='top', fontsize=10, fontweight='bold')
ax2.text(3, ax2.get_ylim()[1] * 0.95, 'b', ha='center', va='top', fontsize=10, fontweight='bold')

import scipy.stats as st
st.ttest_rel(recovery_df['inner_recovery'], recovery_df['rural_recovery'])

# load Landsat TAC
relative_path = '/2_Output/tac_landsat_city_3zones_exclude_disturbance'+sd_name+'.npz'
TACs = np.load(current_dir+relative_path)['array1'] # v5,v4.2
TACs_mean = np.nanmean(TACs, axis=2)
ID = np.load(current_dir+relative_path)['array2'] # v5,v4.2
df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_edge','rural_bgr']
df_tac['ID'] = ID

df_merge = pd.merge(recovery_df,df_tac,on='ID')
tac_uc = df_merge['urban_core']
tac_ue = df_merge['urban_edge']
tac_rb = df_merge['rural_bgr']

rt_uc = df_merge['inner_rt']
rt_ue = df_merge['sub_rt']
rt_rb = df_merge['rural_rt']

rc_uc = df_merge['inner_recovery']
rc_ue = df_merge['sub_recovery']
rc_rb = df_merge['rural_recovery']

dt_vi_uc = df_merge['dt_vi_inner']
dt_vi_ue = df_merge['dt_vi_sub']
dt_vi_rb = df_merge['dt_vi_rural']
print(st.linregress(tac_uc,rt_uc))
print(st.linregress(tac_uc,rc_uc))
print(st.linregress(tac_uc,dt_vi_uc))

slope00, intercept00, r_value00, p_value00, std_err00 = st.linregress(rt_uc, rc_uc)
slope01, intercept01, r_value01, p_value01, std_err01 = st.linregress(rt_ue, rc_ue)
slope02, intercept02, r_value02, p_value02, std_err02 = st.linregress(rt_rb, rc_rb)

ax3.scatter(rt_uc,rc_uc,color=colors[0], s=18, alpha=0.1)
ax3.scatter(rt_ue,rc_ue,color=colors[1], s=18, alpha=0.1)
ax3.scatter(rt_rb,rc_rb,color=colors[2], s=18, alpha=0.1)
x_fit00 = np.array([rt_uc.min(), rt_uc.max()])
ax3.plot(x_fit00, slope00 * x_fit00 + intercept00, colors[0], linewidth=1.5)
x_fit01 = np.array([rt_ue.min(), rt_ue.max()])
ax3.plot(x_fit01, slope01 * x_fit01 + intercept01, colors[1], linewidth=1.5)
x_fit02 = np.array([rt_rb.min(), rt_rb.max()])
ax3.plot(x_fit02, slope02 * x_fit02 + intercept02, 'k', linewidth=1.5)
ax3.set_xlabel('Resistance')
ax3.set_ylabel('Recovery')

slope10, intercept10, r_value10, p_value10, std_err10 = st.linregress(tac_uc, rt_uc)
slope11, intercept11, r_value11, p_value11, std_err11 = st.linregress(tac_ue, rt_ue)
slope12, intercept12, r_value12, p_value12, std_err12 = st.linregress(tac_rb, rt_rb)

ax4.scatter(tac_uc, rt_uc, color=colors[0], s=18, alpha=0.1)
ax4.scatter(tac_ue, rt_ue, color=colors[1], s=18, alpha=0.1)
ax4.scatter(tac_rb, rt_rb, color=colors[2], s=18, alpha=0.1)

x_fit10 = np.array([tac_uc.min(), tac_uc.max()])
x_fit11 = np.array([tac_ue.min(), tac_ue.max()])
x_fit12 = np.array([tac_rb.min(), tac_rb.max()])
ax4.plot(x_fit10, slope10 * x_fit10 + intercept10, colors[0], linewidth=1.5)
ax4.plot(x_fit11, slope11 * x_fit11 + intercept11, colors[1], linewidth=1.5)
ax4.plot(x_fit12, slope12 * x_fit12 + intercept12, 'k', linewidth=1.5)
ax4.set_xlabel('TAC$_{ED}$')
ax4.set_ylabel('Resistance')

# Second subplot: TAC vs Recovery
slope20, intercept20, r_value20, p_value20, std_err20 = st.linregress(tac_uc, rc_uc)
slope21, intercept21, r_value21, p_value21, std_err21 = st.linregress(tac_ue, rc_ue)
slope22, intercept22, r_value22, p_value22, std_err22 = st.linregress(tac_rb, rc_rb)

ax5.scatter(tac_uc, rc_uc, color=colors[0], s=18, alpha=0.1)
ax5.scatter(tac_ue, rc_ue, color=colors[1], s=18, alpha=0.1)
ax5.scatter(tac_rb, rc_rb, color=colors[2], s=18, alpha=0.1)

ax5.plot(x_fit10, slope20 * x_fit10 + intercept20, colors[0], linewidth=1.5)
ax5.plot(x_fit11, slope21 * x_fit11 + intercept21, colors[1], linewidth=1.5)
ax5.plot(x_fit12, slope22 * x_fit12 + intercept22, 'k-', linewidth=1.5)


ax5.set_xlabel('TAC$_{ED}$')
ax5.set_ylabel('Recovery')


slope30, intercept30, r_value30, p_value30, std_err30 = st.linregress(tac_uc, dt_vi_uc)
slope31, intercept31, r_value31, p_value31, std_err31 = st.linregress(tac_ue, dt_vi_ue)
slope32, intercept32, r_value32, p_value32, std_err32 = st.linregress(tac_rb, dt_vi_rb)

x_fit = np.array([tac_uc.min(), tac_uc.max()])
ax6.scatter(tac_uc,dt_vi_uc,color=colors[0], s=18, alpha=0.1)
ax6.scatter(tac_ue,dt_vi_ue,color=colors[1], s=18, alpha=0.1)
ax6.scatter(tac_rb,dt_vi_rb,color=colors[2], s=18, alpha=0.1)

ax6.plot(x_fit10, slope30 * x_fit10 + intercept30, colors[0], linewidth=1.5)
ax6.plot(x_fit11, slope31 * x_fit11 + intercept31, colors[1], linewidth=1.5)
ax6.plot(x_fit12, slope32 * x_fit12 + intercept32, 'k', linewidth=1.5)


ax6.set_xlabel('TAC$_{ED}$')
ax6.set_ylabel('ΔVI')

plt.tight_layout()
plt.savefig(os.path.join(current_dir, '4_Figures', 'recovery_resistance_TAC.png'), dpi=900)
plt.show()

# Perform statistical test
print("T-test results (inner vs rural recovery):")
print(st.ttest_rel(recovery_df['rural_rt'], recovery_df['sub_rt']))
print(st.ttest_rel(recovery_df['inner_rt'], recovery_df['sub_rt']))
print(st.ttest_rel(recovery_df['inner_rt'], recovery_df['rural_rt']))

print(st.ttest_rel(recovery_df['rural_recovery'], recovery_df['sub_recovery']))
print(st.ttest_rel(recovery_df['inner_recovery'], recovery_df['sub_recovery']))
print(st.ttest_rel(recovery_df['inner_recovery'], recovery_df['rural_recovery']))