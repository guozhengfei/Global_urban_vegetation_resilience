import numpy as np
import tifffile as tf
import matplotlib; matplotlib.use('Qt5Agg')
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

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    res_folder = current_dir + '/2_Output/tac_Landsat/'
    urban_folder = current_dir + '/2_Output/nanFrac_urban_label_Landsat/'
    folder = current_dir + '/2_Output/VI_Landsat/'
    filenames = os.listdir(res_folder)
    names = []
    for name in filenames:
        if name.startswith('res_move') and name.endswith('.npy'):
            names.append(name)
    IDs = []
    for name in names:
        id = name.split('_')[-1].split('.npy')[0]
        IDs.append(float(id))
    IDs = np.sort(IDs)

    RES = []
    recovery_stats = []  # List to store recovery statistics
    IDs_num = []
    for id in IDs:
        res = np.load(res_folder + 'res_move_ave__' + str(id) + '.npy') # res
        df_vi = pd.read_csv(folder+'NDVI_8d_'+str(id)+'.csv').astype(float)
        vis = df_vi.iloc[:,:-2].values
        vis_fill = fill_with_climatology(vis,24,12)
        vis_mean = np.nanmean(vis,axis=1)

        urban_nonfrac_labels = np.load(urban_folder+'label_'+str(id)+'.npy')
        urban_lab = urban_nonfrac_labels[0,:]
        nan_frac = urban_nonfrac_labels[1,:]

        res_tmp_all = np.nanmean(res, axis=0)
        res_tmp_inner = np.nanmean(res[urban_lab == 2,:], axis=0)
        vi_inner_mean = np.mean(vis_mean[urban_lab == 2])
        res_tmp_sub = np.nanmean(res[urban_lab == 1, :], axis=0)
        vi_sub_mean = np.mean(vis_mean[urban_lab == 1])
        res_tmp_rural = np.nanmean(res[urban_lab == 0, :], axis=0)
        vi_rural_mean = np.mean(vis_mean[urban_lab == 0])

        # plt.figure(); plt.plot(vi_inner_mean)

        # Find indices of local minima
        from scipy.signal import argrelextrema
        local_min_indices = argrelextrema(res_tmp_all, np.less)[0]
        
        # Get values at local minima
        local_min_values = res_tmp_all[local_min_indices]
        
        # Sort indices by values and get top 10 deepest minima
        top_10_min_idx = local_min_indices[np.argsort(local_min_values)[:3]]
        
        # Sort the indices in chronological order
        min_idx = np.sort(top_10_min_idx)
        min_idx = np.array([x for x in min_idx if x < 286])

        # Fit exponential recovery curves for each zone
        zones = [res_tmp_inner, res_tmp_sub, res_tmp_rural]
        zone_names = ['inner', 'sub', 'rural']
        zone_stats = {
            'ID': id,
            'inner_rt': abs(vi_inner_mean/np.nanmean(res_tmp_inner[min_idx])),  # Add minimum res for inner zone
            'sub_rt': abs(vi_sub_mean/np.nanmean(res_tmp_sub[min_idx])),      # Add minimum res for sub zone
            'rural_rt': abs(vi_rural_mean/np.nanmean(res_tmp_rural[min_idx]))    # Add minimum res for rural zone
        }

        for zone_name, zone_data in zip(zone_names, zones):

            recovery = abs(zone_data[min_idx]/zone_data[min_idx+2]).mean()
            zone_stats[f'{zone_name}_recovery'] = recovery
        
        recovery_stats.append(zone_stats)

        print(id)

    # Convert recovery statistics to DataFrame
    recovery_df = pd.DataFrame(recovery_stats)
    
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

    # Columns to check for outliers
    columns_to_check = ['inner_rt', 'sub_rt', 'rural_rt', 
                       'inner_recovery', 'sub_recovery', 'rural_recovery']
    
    # Remove outliers
    recovery_df = remove_outliers(recovery_df, columns_to_check)
    
    # Drop any remaining NaN values
    recovery_df = recovery_df.dropna()

    # Save recovery statistics
    # output_path = os.path.join(current_dir, '2_Output', 'recovery_statistics_no_outliers.csv')
    # recovery_df.to_csv(output_path, index=False)
    # print(f"Recovery statistics saved to: {output_path}")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8*0.72, 2.8*0.65))
    
    # Plot minimum res values
    min_means = recovery_df[['inner_rt', 'sub_rt', 'rural_rt']].mean()
    min_sems = recovery_df[['inner_rt', 'sub_rt', 'rural_rt']].std() * 0.2
    
    x = np.arange(3)
    width = 0.35
    
    ax1.bar(x, min_means, width, yerr=min_sems,color=['#7fc97f', '#beaed4', '#fdc086'])
    # ax1.set_ylim([-0.062,-0.035])
    ax1.set_xticks(x)
    ax1.set_ylim([27,38])
    ax1.set_xticklabels(['UC', 'UE', 'UR'])
    ax1.set_ylabel('Resistance')

    # Plot recovery times
    rec_means = recovery_df[['inner_recovery', 'sub_recovery', 'rural_recovery']].mean()
    rec_sems = recovery_df[['inner_recovery', 'sub_recovery', 'rural_recovery']].std() * 0.1
    
    ax2.bar(x, rec_means, width, yerr=rec_sems,color=['#7fc97f', '#beaed4', '#fdc086'])
    # ax2.set_ylim([3.1,3.5])

    ax2.set_xticks(x)
    ax2.set_xticklabels(['Urban_core', 'Urban_edge', 'Rural_bgr'])
    ax2.set_ylabel('Recovery')
    ax2.set_ylim([1.8, 2.2])
    plt.tight_layout()
    
    # Save the figure
    # fig_path = os.path.join(current_dir, '4_Figures', 'recovery_analysis_no_outliers.png')
    # plt.savefig(fig_path, dpi=900, bbox_inches='tight')
    plt.show()

    # Perform statistical test
    print("T-test results (inner vs rural recovery):")
    print(st.ttest_rel(recovery_df['rural_rt'], recovery_df['sub_rt']))