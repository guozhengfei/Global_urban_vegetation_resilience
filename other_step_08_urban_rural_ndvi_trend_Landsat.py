import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
# import matplotlib;
# matplotlib.use('Qt5Agg')
import scipy.signal as ss
import multiprocess as mp
import os
import warnings
import cv2
warnings.filterwarnings("ignore")
import numpy as np


if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    tac_folder = current_dir + '/2_Output/VI_trend/'
    urban_folder = current_dir + '/2_Output/nanFrac_urban_label_Landsat/'

    filenames = os.listdir(tac_folder)
    names = []
    for name in filenames:
        if name.startswith('V') and name.endswith('.npy'): #'.npy'
            names.append(name)
    IDs = []
    for name in names:
        id = name.split('_')[-1].split('.npy')[0]
        IDs.append(float(id))
    IDs = np.sort(IDs)

    TAC = []
    # TAC_TMP = []
    IDs_num = []
    for id in IDs:
        tac = np.load(tac_folder + 'VI_trendndvi_trend_' + str(id) + '.npy')
        tac_mean_spa = tac
        urban_nonfrac_labels = np.load(urban_folder+'label_'+str(id)+'.npy')
        urban_lab = urban_nonfrac_labels[0,:]
        nan_frac = urban_nonfrac_labels[1,:]
        # plt.close()
        # plt.figure(); plt.hist(nan_frac,50)
        tac_inner = np.nanmean(tac_mean_spa[urban_lab == 2])
        tac_sub = np.nanmean(tac_mean_spa[urban_lab == 1])
        tac_rural = np.nanmean(tac_mean_spa[urban_lab == 0])
        TAC.append([tac_inner, tac_sub, tac_rural])


        IDs_num.append(float(id))
        print(id)
    ndvi_trend = np.array(TAC)

    IDs_num = np.array(IDs_num)
    import pandas as pd
    df_trends = pd.DataFrame(ndvi_trend, 
                           columns=['uc_vi_trend', 'ue_vi_trend', 'ru_vi_trend'])
    df_trends['ID'] = IDs
    
    # Calculate trend differences
    df_trends['vi_trend_diff'] = df_trends['uc_vi_trend'] - df_trends['ru_vi_trend']
    
    # Save to CSV
    output_csv = os.path.join(current_dir, '2_Output', 'vi_trends.csv')
    df_trends.to_csv(output_csv, index=False)
    
    # tac_trend
    df_tac = pd.read_csv(current_dir + '/2_Output/tac_trends.csv')

    # Merge dataframes on 'ID'
    merged_df = pd.merge(df_trends, df_tac, on='ID')
    
    # Create correlation heatmap
    import seaborn as sns
    
    # Select all columns except 'ID'
    cols_for_corr = [col for col in merged_df.columns if col != 'ID']
    
    # Calculate correlation matrix
    corr_matrix = merged_df[cols_for_corr].corr()
    
    # Create figure with larger size
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True,          # Show correlation values
                fmt='.2f',           # Format to 2 decimal places
                cmap='RdBu_r',       # Red-Blue diverging colormap
                center=0,            # Center colormap at 0
                square=True,         # Make cells square
                cbar_kws={'label': 'Correlation Coefficient'},
                vmin=-1, vmax=1)     # Set correlation range
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure
    # plt.savefig(os.path.join(current_dir, '4_Figures', 'trend_correlations_heatmap.png'), 
    #             dpi=300, bbox_inches='tight')
    plt.show()




