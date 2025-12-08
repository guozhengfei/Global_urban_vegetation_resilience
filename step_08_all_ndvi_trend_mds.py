import numpy as np
import tifffile as tf
import matplotlib;
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.signal as ss
import multiprocess as mp
import os
import warnings

warnings.filterwarnings("ignore")
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

def IQR_filter2(array):
    p25 = np.nanpercentile(array, 25,axis=1)
    p75 = np.nanpercentile(array,75,axis=1)
    IQR = p75-p25
    maxV = np.tile(p75 + 0.5*IQR, (array.shape[1], 1)).T
    minV = np.tile(p25 - 0.5 * IQR, (array.shape[1], 1)).T
    array[array < minV] = np.nan
    array[array > maxV] = np.nan
    arraynew = array
    return arraynew


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


# Add the fill_nan_with_climatology function
def fill_nan_with_climatology(EVI, bands_year=12):
    num_years = EVI.shape[1] // bands_year
    climatology = np.zeros((EVI.shape[0], EVI.shape[1]))
    
    for year in range(num_years):
        if year < 2:
            start = 0
            end = 5 * bands_year
        elif year > num_years - 3:
            start = (num_years - 5) * bands_year
            end = num_years * bands_year
        else:
            start = (year - 2) * bands_year
            end = (year + 3) * bands_year
        
        climatology[:, year * bands_year:(year + 1) * bands_year] = np.nanmean(
            EVI[:, start:end].reshape(EVI.shape[0], 5, bands_year), axis=1
        )
    
    EVI_filled = np.where(np.isnan(EVI), climatology, EVI)
    
    return EVI_filled

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/nadir_ndvi_15d_500/'
    grass_folder = current_dir + '/1_Input/grassCover_250m/'
    tree_folder = current_dir + '/1_Input/treeCover_250m/'
    crop_folder = current_dir + '/1_Input/cropCover_250m/'
    water_folder = current_dir + '/1_Input/waterCover_250m/'
    urbanExp_folder = current_dir + '/1_Input/urban_expansion_frac_500m/'
    urban_folder_2018 = current_dir + '/1_Input/urban/urban_2018_1000m/'
    urban_folder_1990 = current_dir + '/1_Input/urban/urban_1990_250m/'
    # urban_folder = current_dir+'/1_Input/urban_area/'

    IDs = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array2']
    IDs = np.sort(IDs)

    TAC = []
    for id in IDs:#IDs:
        EVI0 = tf.imread(EVI_folder + 'nadir_ndvi_15d_' + str(id) + '.tif')[:, :, :264*2][:,:,::2]
        nan_frac = np.sum(np.isnan(EVI0),axis=2)/EVI0.shape[2]
        crop_frc = tf.imread(crop_folder + 'cropC_' + str(id) + '.tif')
        crop_frc = cv2.resize(crop_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)

        grass_frc = tf.imread(grass_folder + 'grassC_' + str(id) + '.tif')
        grass_frc = cv2.resize(grass_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)

        tree_frc = tf.imread(tree_folder + 'treeC_' + str(id) + '.tif')
        tree_frc = cv2.resize(tree_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)

        water_frc = tf.imread(water_folder + 'waterC_' + str(id) + '.tif')
        water_frc = cv2.resize(water_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)

        urbanExp_frc = tf.imread(urbanExp_folder + 'urban_exp_C_' + str(id) + '.tif')

        crop_frc[crop_frc > 0.2] = np.nan
        veg_nature = grass_frc + tree_frc
        crop_frc[crop_frc > veg_nature] = np.nan
        mask_miss = np.sum(np.isnan(EVI0), axis=2) / EVI0.shape[2]
        mask = (np.isnan(crop_frc) | (mask_miss > 0.3) | (water_frc>0.3) | (urbanExp_frc>0.2)).astype(float)
        urban_1990 = tf.imread(urban_folder_1990 + 'urban_1990_' + str(id) + '.tif').astype(float)
        urban_1990 = cv2.resize(urban_1990, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_NEAREST)
        urban_1990[urban_1990 == 0] = np.nan;

        urban_2018 = tf.imread(urban_folder_2018 + 'fvc_' + str(id) + '.tif').astype(float)
        urban_2018_rsz = cv2.resize(urban_2018, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_NEAREST)
        rural_near = urban_2018_rsz * 1  # rural-urban interface
        for i in range(3 * 2):
            rural_near = extend_edge(rural_near)

        rural_bgr = rural_near * 1  # rural background
        for i in range(10 * 2):
            rural_bgr = extend_edge(rural_bgr)
        rural_near_v2 = rural_near * 1
        rural_near_v2[np.isnan(rural_near_v2)] = 0
        rural_near_v2[rural_near_v2 != 0] = np.nan
        urban_rural_bgr = rural_bgr + rural_near_v2
        urban_rural_bgr[urban_rural_bgr==0]=np.nan

        #urban_1990[mask==1] = np.nan
        EVI_uc = np.nanmean(EVI0[~np.isnan(urban_1990),:],axis=0)
        EVI_uc_yearly = EVI_uc.reshape((22, 12))
        EVI_uc_yearly = np.nanmean(EVI_uc_yearly, axis=1)

        #urban_rural_bgr[mask==1] = np.nan
        EVI_ru = np.nanmean(EVI0[~np.isnan(urban_rural_bgr), :], axis=0)
        EVI_ru_yearly = EVI_ru.reshape((22, 12))
        EVI_ru_yearly = np.nanmean(EVI_ru_yearly, axis=1)

        TAC.append([ id, cal_slope(EVI_uc_yearly), cal_slope(EVI_ru_yearly), cal_slope(EVI_uc_yearly)-cal_slope(EVI_ru_yearly)])

        print(id)
    ndvi_trend = np.array(TAC)
    import pandas as pd

    df_trends = pd.DataFrame(ndvi_trend,
                             columns=['ID', 'uc_vi_trend', 'ru_vi_trend', 'vi_trend_diff'])

    # Save to CSV
    output_csv = os.path.join(current_dir, '2_Output', 'vi_trends_mds.csv')
    df_trends.to_csv(output_csv, index=False)

    # tac_trend
    df_tac = pd.read_csv(current_dir + '/2_Output/tac_trends_modis.csv')

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
    # plt.savefig(os.path.join(current_dir, '4_Figures', 'trend_correlations_heatmap.png'),
    #             dpi=300, bbox_inches='tight')
    plt.show()