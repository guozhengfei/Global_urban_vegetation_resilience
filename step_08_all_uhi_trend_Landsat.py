import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib;
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.ndimage import convolve1d
import os
import multiprocess as mp
import scipy.stats as st

def cal_slope(y):
    # Create x array (years)
    x = np.arange(len(y))
    
    # Remove NaN values
    mask = ~np.isnan(y)
    if np.sum(mask) < 3:  # Need at least 3 points for meaningful trend
        return np.nan
    
    x_valid = x[mask]
    y_valid = y[mask]
    
    try:
        # Calculate linear regression
        A = np.vstack([x_valid, np.ones(len(x_valid))]).T
        slope, _ = np.linalg.lstsq(A, y_valid, rcond=None)[0]
        return slope
    except:
        return np.nan


if __name__ == '__main__':

    uhi_day = pd.read_csv('/Volumes/Zhengfei_01/urban_env_data/uhi_day_csv_urban_751.csv')
    uhi_night = pd.read_csv('/Volumes/Zhengfei_01/urban_env_data/uhi_night_csv_urban_751.csv')
    uhi = (uhi_day.values[:, 1:-4] + uhi_night.values[:, 1:-4]) / 2
    uhi_trends = []
    for i in range(uhi.shape[0]):
        uhi_trends.append(st.linregress(range(16), uhi[i, :]).slope)
    uhi_day['uhi_trend'] = np.array(uhi_trends)
    uhi_trends_df = uhi_day[['ID', 'uhi_trend']]
    uhi_trends_df.to_csv('/Volumes/Zhengfei_01/project 3 urban resilience/2_Output/uhi_trends.csv')


    # current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\', '/')
    # folder = os.path.join('..','..','urban_env_data','uhi_1000m')
    # filenames = os.listdir(folder)
    # urban_folder_1990 = current_dir + '/project 3 urban resilience/1_Input/urban/urban_1990_250m/'
    #
    # names = []
    # for name in filenames:
    #     if name.startswith('u'):
    #         names.append(name)
    #
    # IDs = []
    # for name in names:
    #     id = float(name.split('_')[-1].split('.tif')[0])
    #     IDs.append(id)
    #
    # IDs = np.sort(IDs)
    # trends = []
    # for id in IDs:
    #     uhi_i = tf.imread(folder + '/uhi_' + str(id) + '.tif')
    #     urban_1990 = tf.imread(urban_folder_1990 + 'urban_1990_' + str(id) + '.tif').astype(float)
    #     urban_1990 = cv2.resize(urban_1990, (uhi_i.shape[1], uhi_i.shape[0]), interpolation=cv2.INTER_NEAREST)
    #     urban_1990[urban_1990 == 0] = np.nan
    #     urban_1990[~np.isnan(urban_1990)] = 1
    #     uhi_i_uc = uhi_i[~np.isnan(urban_1990),:]
    #     uhi_i_uc_mean = np.nanmean(uhi_i_uc,axis=0)
    #     # plt.figure(); plt.imshow(gdp_i[:,:,-1]); plt.show()
    #
    #     trends.append([float(id),cal_slope(uhi_i_uc_mean)])
    #     print(id)
    #
    #
    # trends_arr = np.array(trends)
    # import pandas as pd
    # df_trends = pd.DataFrame(trends_arr,
    #                          columns=['ID','uhi_trend'])
    #
    # output_csv = os.path.join(current_dir, 'project 3 urban resilience', '2_Output', 'uhi_trends.csv')
    # df_trends.to_csv(output_csv, index=False)
    #
    # # tac_trend
    # df_tac = pd.read_csv(current_dir + '/project 3 urban resilience/2_Output/tac_trends.csv')
    #
    # # Merge dataframes on 'ID'
    # merged_df = pd.merge(df_trends, df_tac, on='ID')
    # import seaborn as sns
    # cols_for_corr = [col for col in merged_df.columns if col != 'ID']
    # corr_matrix = merged_df[cols_for_corr].corr()
    #
    # # Create figure with larger size
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr_matrix,
    #             annot=True,  # Show correlation values
    #             fmt='.2f',  # Format to 2 decimal places
    #             cmap='RdBu_r',  # Red-Blue diverging colormap
    #             center=0,  # Center colormap at 0
    #             square=True,  # Make cells square
    #             cbar_kws={'label': 'Correlation Coefficient'},
    #             vmin=-1, vmax=1)  # Set correlation range
    #
    # # Rotate x-axis labels for better readability
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=0)
    #
    # # Adjust layout to prevent label cutoff
    # plt.tight_layout()
    #
    # # Save figure
    # # plt.savefig(os.path.join(current_dir, '4_Figures', 'pop_trend_correlations_heatmap.png'),
    # #             dpi=300, bbox_inches='tight')
    # plt.show()
