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
    folder = current_dir + '/urban_env_data/pop_500m/'
    filenames = os.listdir(folder)

    urban_folder_2018 = current_dir + '/project 3 urban resilience/1_Input/urban/urban_2018_1000m/'
    urban_folder_1990 = current_dir + '/project 3 urban resilience/1_Input/urban//urban_1990_250m/'

    names = []
    for name in filenames:
        if name.startswith('n'):
            names.append(name)

    IDs = []
    for name in names:
        id = float(name.split('_')[-1].split('.tif')[0])
        IDs.append(id)

    IDs = np.sort(IDs)
    trends = []
    for id in IDs:
        gdp_i = tf.imread(folder+'ntl_'+str(id)+'.tif').astype(float)
        # plt.figure(); plt.imshow(gdp_i[:,:,-1]); plt.show()
        urban_2018 = tf.imread(urban_folder_2018 + 'fvc_' + str(id) + '.tif').astype(float)
        urban_2018_rsz = cv2.resize(urban_2018, (gdp_i.shape[1], gdp_i.shape[0]),cv2.INTER_NEAREST)
        urban_2018_rsz[urban_2018_rsz != float(id)] = np.nan
        urban_1990 = tf.imread(urban_folder_1990 + 'urban_1990_' + str(id) + '.tif').astype(float)
        urban_1990= cv2.resize(urban_1990, (gdp_i.shape[1], gdp_i.shape[0]),cv2.INTER_NEAREST)
        urban_1990[urban_1990 == 0] = np.nan

        rural_near = urban_2018_rsz * 1  # rural-urban interface
        for i in range(3):
            rural_near = extend_edge(rural_near)

        rural_bgr = rural_near * 1  # rural background
        for i in range(10):
            rural_bgr = extend_edge(rural_bgr)

        # URBAN 1990
        gdp_uc = gdp_i[~np.isnan(urban_1990),:]
        gdp_ru = gdp_i[~np.isnan(rural_bgr), :]

        gdp_uc_mean = np.nanmean(gdp_uc,axis=0)
        gdp_ru_mean = np.nanmean(gdp_ru, axis=0)

        gdp_uc_trend = cal_slope(gdp_uc_mean)
        gdp_ru_trend = cal_slope(gdp_ru_mean)

        gpd_trend_diff = gdp_uc_trend-gdp_ru_trend

        trends.append([float(id),gdp_uc_trend, gdp_ru_trend, gpd_trend_diff])
        print(id)


    trends_arr = np.array(trends)
    import pandas as pd
    df_trends = pd.DataFrame(trends_arr,
                             columns=['ID','uc_pop_trend', 'ue_pop_trend', 'pop_trend_diff'])

    output_csv = os.path.join(current_dir, 'project 3 urban resilience', '2_Output', 'pop_trends.csv')
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
