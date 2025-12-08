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
import cv2

def extend_edge(array):
    array1 = array * 1
    array1[1:, :] = array[:-1, :]  # Shift elements up
    array2 = array * 1
    array2[:-1, :] = array[1:, :]  # Shift elements down
    array3 = array * 1
    array3[:, 1:] = array[:, :-1]  # Shift elements left
    array4 = array * 1
    array4[:, :-1] = array[:, 1:]  # Shift elements right
    array5 = array * 1
    array5[:-1, 1:] = array[1:, :-1]  # Shift elements up-left
    array6 = array * 1
    array6[:-1, :-1] = array[1:, 1:]  # Shift elements up-right
    array7 = array * 1
    array7[1:, 1:] = array[:-1, :-1]  # Shift elements down-right
    array8 = array * 1
    array8[1:, :-1] = array[:-1, 1:]  # Shift elements down-left
    stacked_array = np.stack([array1, array2, array3, array4, array5, array6, array7, array8], axis=0)
    result = np.nanmean(stacked_array, axis=0)
    result[result > 0] = 1
    return result

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
    current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\', '/')
    folder = os.path.join('..','..','urban_env_data','shortwave')
    filenames = os.listdir(folder)

    names = []
    for name in filenames:
        if name.startswith('s'):
            names.append(name)

    IDs = []
    for name in names:
        id = float(name.split('_')[-1].split('.tif')[0])
        IDs.append(id)

    IDs = np.sort(IDs)
    trends = []
    for id in IDs:
        sw_i0 = tf.imread(folder + '/sw_' + str(id) + '.tif') / (3600 * 24)
        sw_i = np.nanmean(np.nanmean(sw_i0,axis=0),axis=0)
        sw_i_yearly = np.reshape(sw_i, (int(sw_i.shape[0] / 12),12))
        sw_i_yearly = np.nanmean(sw_i_yearly, axis=1)
        # plt.figure(); plt.imshow(gdp_i[:,:,-1]); plt.show()

        trends.append([float(id),cal_slope(sw_i_yearly)])
        print(id)


    trends_arr = np.array(trends)
    import pandas as pd
    df_trends = pd.DataFrame(trends_arr,
                             columns=['ID','sw_trend'])

    output_csv = os.path.join(current_dir, 'project 3 urban resilience', '2_Output', 'sw_trends.csv')
    df_trends.to_csv(output_csv, index=False)

    # tac_trend
    df_tac = pd.read_csv(current_dir + '/project 3 urban resilience/2_Output/tac_trends.csv')

    # Merge dataframes on 'ID'
    merged_df = pd.merge(df_trends, df_tac, on='ID')
    import seaborn as sns
    cols_for_corr = [col for col in merged_df.columns if col != 'ID']
    corr_matrix = merged_df[cols_for_corr].corr()

    # Create figure with larger size
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix,
                annot=True,  # Show correlation values
                fmt='.2f',  # Format to 2 decimal places
                cmap='RdBu_r',  # Red-Blue diverging colormap
                center=0,  # Center colormap at 0
                square=True,  # Make cells square
                cbar_kws={'label': 'Correlation Coefficient'},
                vmin=-1, vmax=1)  # Set correlation range

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    # plt.savefig(os.path.join(current_dir, '4_Figures', 'pop_trend_correlations_heatmap.png'),
    #             dpi=300, bbox_inches='tight')
    plt.show()
