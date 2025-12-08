import numpy as np
import matplotlib;
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
plt.rc('lines', linewidth=1.05)  # Increase default line width
plt.rc('axes', linewidth=1.05)   # Increase axes line width
plt.rc('grid', linewidth=1.05)   # Increase grid line width
plt.tick_params(width=1.05, labelsize=14)
plt.rc('xtick.major', size=4, width=1.05)    # Increase length and width of major ticks
plt.rc('ytick.major', size=4, width=1.05)    # Increase length and width of major ticks
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
pattern2 = create_pattern(min_val=-1.0, recovery_rate=0.08)*0.1  # Medium recovery
pattern3 = create_pattern(min_val=-1.5, recovery_rate=0.05)*0.1  # Slow recovery

# Create figure
plt.figure(figsize=(8.5*0.7, 3.5*0.7))

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
plt.savefig('../4_Figures/Fig05_conceptual_recovery.png', dpi=600, bbox_inches='tight')
plt.show()


current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
recovery_df = pd.read_csv(os.path.join(current_dir, '2_Output', 'recovery_statistics_no_outliers.csv'))

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8 * 0.72, 2.8 * 0.65))

# Plot minimum TAC values
min_means = recovery_df[['inner_rt', 'sub_rt', 'rural_rt']].mean()
min_sems = recovery_df[['inner_rt', 'sub_rt', 'rural_rt']].std() * 0.2

x = np.arange(3)
width = 0.35

ax1.bar(x, min_means, width, yerr=min_sems, color=['#2166ac', '#67a9cf', '#b2182b'],
        error_kw=dict(lw=1.05, capsize=3, capthick=1.05))
ax1.set_xticks(x)
# ax1.set_ylim([27,38])
ax1.set_xticklabels(['UC', 'UE', 'UR'])
ax1.set_ylabel('Resistance')

# Plot recovery times
rec_means = recovery_df[['inner_recovery', 'sub_recovery', 'rural_recovery']].mean()
rec_sems = recovery_df[['inner_recovery', 'sub_recovery', 'rural_recovery']].std() * 0.1

ax2.bar(x, rec_means, width, yerr=rec_sems, color=['#2166ac', '#67a9cf', '#b2182b'],
        error_kw=dict(lw=1.05, capsize=3, capthick=1.05))

ax2.set_xticks(x)
ax2.set_xticklabels(['UC', 'UE', 'RB'])
ax2.set_ylabel('Recovery')
# ax2.set_ylim([1.8, 2.2])
plt.tight_layout()

# Save the figure
fig_path = os.path.join(current_dir, '4_Figures', 'recovery_analysis_no_outliers.png')
plt.savefig(fig_path, dpi=900, bbox_inches='tight')
plt.show()

import scipy.stats as st
st.ttest_rel(recovery_df['inner_recovery'], recovery_df['rural_recovery'])

# load Landsat TAC
relative_path = '/2_Output/tac_landsat_city_3zones.npz'
TACs = np.load(current_dir+relative_path)['array1'] # v5,v4.2
TACs_mean = np.nanmean(TACs, axis=2)
ID = np.load(current_dir+relative_path)['array2'] # v5,v4.2
df_tac = pd.DataFrame(TACs_mean)
df_tac.columns=['urban_core','urban_edge','rural_bgr']
df_tac['ID'] = ID

df_merge = pd.merge(recovery_df,df_tac,on='ID')
tac_uc = df_merge['urban_core']
rt_uc = df_merge['inner_rt']
rc_uc = df_merge['inner_recovery']
print(st.linregress(tac_uc,rt_uc))
print(st.linregress(tac_uc,rc_uc))

# Calculate linear regression for both relationships
slope1, intercept1, r_value1, p_value1, std_err1 = st.linregress(tac_uc, rt_uc)
slope2, intercept2, r_value2, p_value2, std_err2 = st.linregress(tac_uc, rc_uc)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8 * 0.72, 2.8 * 0.72))

# First subplot: TAC vs Resistance
ax1.scatter(tac_uc, rt_uc, color='#2166ac', s=18, alpha=0.3)
# Add fit line
x_fit = np.array([tac_uc.min(), tac_uc.max()])
ax1.plot(x_fit, slope1 * x_fit + intercept1, 'k-', linewidth=1.05)
ax1.set_xlabel('TAC')
ax1.set_ylabel('Resistance')

# Second subplot: TAC vs Recovery
ax2.scatter(tac_uc, rc_uc, color='#2166ac', s=18, alpha=0.3)
# Add fit line
ax2.plot(x_fit, slope2 * x_fit + intercept2, 'k-', linewidth=1.05)
ax2.set_xlabel('TAC')
ax2.set_ylabel('Recovery')

plt.tight_layout()
plt.savefig(os.path.join(current_dir, '4_Figures', 'recovery_resistance_TAC.png'), dpi=900)
plt.show()

# Perform statistical test
print("T-test results (inner vs rural recovery):")
print(st.ttest_rel(recovery_df['rural_rt'], recovery_df['sub_rt']))