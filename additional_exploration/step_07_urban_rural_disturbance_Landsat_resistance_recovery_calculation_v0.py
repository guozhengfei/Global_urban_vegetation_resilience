import numpy as np
import tifffile as tf
import matplotlib;
from networkx.algorithms.distance_measures import resistance_distance

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
import scipy.signal as ss
import multiprocess as mp
import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import scipy.stats as st

def fill_with_climatology(vis,yr_num,bands_year):
    Evi_sea_rep = np.zeros_like(vis)
    for yr in range(yr_num):
        start_index = (yr - 2) * bands_year
        if start_index < 0: start_index = 0
        end_index = (yr + 3) * bands_year
        if end_index > bands_year * yr_num: end_index = bands_year * yr_num
        data_i = vis[:, start_index:end_index]
        Evi_sea = np.nanmean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)),
                             axis=1)
        Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea
    vis[np.isnan(vis)] = Evi_sea_rep[np.isnan(vis)]
    return vis

# Remove outliers using IQR method
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

if __name__ == '__main__':
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
    sd_names = ['1','1.5','2','2.5']
    for sd_name in sd_names:
        for id in IDs[1:]:
            resis = np.load(current_dir+'/2_Output/Landsat_recovery_resistance/resistance_'+str(id)+ '_smith_'+sd_name+'sd.npy')
            resil = np.load(current_dir+'/2_Output/Landsat_recovery_resistance/resilience_'+str(id)+ '_smith_'+sd_name+'sd.npy')
            # resis = np.load(current_dir + '/2_Output/Landsat_recovery_resistance/resistance_' + str(id) + '_forizier.npy')
            # resil = np.load(current_dir + '/2_Output/Landsat_recovery_resistance/resilience_' + str(id) + '_forizier.npy')
            urban_nonfrac_labels = np.load(urban_folder+'label_'+str(id)+'.npy')
            urban_lab = urban_nonfrac_labels[0,:]

            resis_inner = np.nanmean(resis[urban_lab == 2], axis=0)
            resis_sub = np.nanmean(resis[urban_lab == 1], axis=0)
            resis_rural = np.nanmean(resis[urban_lab == 0], axis=0)
            resil_inner = np.nanmean(resil[urban_lab == 2], axis=0)
            resil_sub = np.nanmean(resil[urban_lab == 1], axis=0)
            resil_rural = np.nanmean(resil[urban_lab == 0], axis=0)

            recovery_stats.append([id,resis_inner,resis_sub,resis_rural,resil_inner,resil_sub,resil_rural])
        recovery_stats_arr = np.array(recovery_stats)

        np.nanmean(recovery_stats_arr,axis=0)
        recovery_df = pd.DataFrame(recovery_stats_arr)
        recovery_df.columns = ['ID','inner_rt','sub_rt','rural_rt','inner_recovery','sub_recovery','rural_recovery']
        columns_to_check = ['inner_rt','sub_rt','rural_rt','inner_recovery','sub_recovery','rural_recovery']
        recovery_df = remove_outliers(recovery_df, columns_to_check)
        recovery_df = recovery_df.dropna()

        # Create figure with two subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8*0.72, 6*0.65))
        # Data for plots
        resistance_data = [recovery_df['inner_rt'], recovery_df['sub_rt'], recovery_df['rural_rt']]
        recovery_data = [recovery_df['inner_recovery'], recovery_df['sub_recovery'], recovery_df['rural_recovery']]
        colors = ['#2166ac', '#67a9cf', '#b2182b']
        labels = ['UC', 'UE', 'RB']

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
        ax1.set_ylim(recovery_df['rural_rt'].min()*1.1,0)
        ax1.text(1, ax1.get_ylim()[1] * 0.95, 'a', ha='center', va='top', fontsize=10)
        ax1.text(2, ax1.get_ylim()[1] * 0.95, 'b', ha='center', va='top', fontsize=10)
        ax1.text(3, ax1.get_ylim()[1] * 0.95, 'c', ha='center', va='top', fontsize=10)

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
        ax2.set_ylim(0,recovery_df['rural_recovery'].max()*1.15)
        ax2.text(1, ax2.get_ylim()[1] * 0.95, 'a', ha='center', va='top', fontsize=10)
        ax2.text(2, ax2.get_ylim()[1] * 0.95, 'b', ha='center', va='top', fontsize=10)
        ax2.text(3, ax2.get_ylim()[1] * 0.95, 'c', ha='center', va='top', fontsize=10)
        plt.tight_layout()

        # load Landsat TAC
        relative_path = '/2_Output/tac_landsat_city_3zones.npz'
        TACs = np.load(current_dir + relative_path)['array1']  # v5,v4.2
        TACs_mean = np.nanmean(TACs, axis=2)
        ID = np.load(current_dir + relative_path)['array2']  # v5,v4.2
        df_tac = pd.DataFrame(TACs_mean)
        df_tac.columns = ['urban_core', 'urban_edge', 'rural_bgr']
        df_tac['ID'] = ID

        df_merge = pd.merge(recovery_df, df_tac, on='ID')
        tac_uc = df_merge['urban_core']
        rt_uc = df_merge['inner_rt']
        rc_uc = df_merge['inner_recovery']

        tac_ue = df_merge['urban_edge']
        rt_uc = df_merge['inner_rt']
        rc_uc = df_merge['inner_recovery']

        tac_uc = df_merge['urban_core']
        rt_uc = df_merge['inner_rt']
        rc_uc = df_merge['inner_recovery']

        # Calculate linear regression for both relationships
        import scipy.stats as st
        slope1, intercept1, r_value1, p_value1, std_err1 = st.linregress(tac_uc, rt_uc)
        slope2, intercept2, r_value2, p_value2, std_err2 = st.linregress(tac_uc, rc_uc)
        print(st.linregress(tac_uc, rt_uc))
        print(st.linregress(tac_uc, rc_uc))

        # Create figure with two subplots
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8 * 0.72, 2.8 * 0.72))

        # First subplot: TAC vs Resistance
        ax3.scatter(tac_uc, rt_uc, color='#2166ac', s=18, alpha=0.2)
        # Add fit line
        x_fit = np.array([tac_uc.min(), tac_uc.max()])
        ax3.plot(x_fit, slope1 * x_fit + intercept1, 'k-', linewidth=1.05)
        ax3.set_xlabel('TAC')
        ax3.set_ylabel('Resistance')

        # Second subplot: TAC vs Recovery
        ax4.scatter(tac_uc, rc_uc, color='#2166ac', s=18, alpha=0.3)
        # Add fit line
        ax4.plot(x_fit, slope2 * x_fit + intercept2, 'k-', linewidth=1.05)
        ax4.set_xlabel('TAC')
        ax4.set_ylabel('Recovery')
        plt.tight_layout()

        sd_name = str(int(float(sd_name)*10))

        fig_path = os.path.join(current_dir, '4_Figures', 'recovery_analysis_sd'+sd_name)
        plt.savefig(fig_path, dpi=900, bbox_inches='tight')
