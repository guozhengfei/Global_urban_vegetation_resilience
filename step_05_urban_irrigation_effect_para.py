import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st
import warnings

# Configure warning filters
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in true_divide")
import os
import tifffile as tf
import cv2
from joblib import Parallel, delayed

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

def cal_H(Td,Tc,Ta,ws_i):
    # calculate LE
    k = 0.4  # von Kármán constant
    hc = 10  # m
    z0 = hc * 0.1  # Norman
    ustar = ws_i * k / (np.log(10 / z0))
    ra = ws_i / ustar ** 2 + 2 / (k * ustar)  # Eq.(2 - 4) in Ryu et al 2008
    ra[ra > 100] = 100
    # ra = rhoa*Cp/H*(df['ts_tree_lds']-df['ta_tree'])
    Pa = 101 * 1000  # Pa
    rhoa = Pa / (287.05 * (Tc + 273.15));  # presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)

    mv_ma = 0.622;  # [-] (Wiki)
    ea = 2.1718e10 * np.exp(-4157. / (Td - 33.91));  # [Pa] (Henderson-Sellers, 1984)
    q = (mv_ma * ea) / (Pa - 0.378 * ea);

    # specific heat of dry air
    Cpd = 1005 + ((Ta + 273.15) - 250) ** 2 / 3364;  # [J kg-1 K-1] (Garratt, 1994)
    # specific heat of air
    Cp = Cpd * (1 + 0.84 * q);
    H = rhoa * Cp * (Tc - Ta) / ra
    return H

def process_id(id):
    try:
        lst = tf.imread(lst_dir_path + '/Landsat_' + str(id) + '.tif')
    except FileNotFoundError:
        return None
    lst = np.nanmean(lst, axis=2)
    treeC = tf.imread(treeC_folder + '/cropC_' + str(id) + '.tif')
    grassC = tf.imread(grassC_folder + '/grassC_' + str(id) + '.tif')
    ndvi = tf.imread(ndvi_folder + '/Landsat_ndvi_' + str(id) + '.tif')
    vegC = treeC + grassC
    urbanExp_frc = tf.imread(urbanExp_folder + '/urban_exp_C_' + str(id) + '.tif')
    urbanExp_frc = cv2.resize(urbanExp_frc, (lst.shape[1], lst.shape[0]), cv2.INTER_LINEAR)
    try:
        ta_BG = tf.imread(ta_MDS_folder + '/TairMax_' + str(id) + '.tif')
    except FileNotFoundError:
        return None
    ta_BG = np.nanmean(ta_BG, axis=2)
    ta_BG = cv2.resize(ta_BG, (lst.shape[1], lst.shape[0]), cv2.INTER_LINEAR)

    ratios_i = ratios[ratios['ID'] == id]
    SW = np.nanmean(tf.imread(sw_folder + '/sw_' + str(id) + '.tif'), axis=2) / (3600 * 24) * ratios_i['SW_ratio'].values  # ratio for calculating noon time values
    SW = cv2.resize(SW, (lst.shape[1], lst.shape[0]), cv2.INTER_LINEAR)
    LW = np.nanmean(tf.imread(lw_folder + '/lw_' + str(id) + '.tif'), axis=2) / (3600 * 24) * ratios_i['LW_ratio'].values
    LW = cv2.resize(LW, (lst.shape[1], lst.shape[0]), cv2.INTER_LINEAR)
    Alb = tf.imread(alb_folder + '/albedo_' + str(id) + '.tif')
    Alb = cv2.resize(Alb, (lst.shape[1], lst.shape[0]), cv2.INTER_LINEAR)

    # Rn = (1-alb)*SW_d+emis*LW_d - sig*emis*Ts**4
    Rn = (1 - 0.001 * Alb) * SW + 0.97 * LW - 0.97 * 5.67037442 * 10 ** -8 * (lst + 273.15) ** 4
    Q = tf.imread(ahe_folder + '/AHE_' + str(id) + '.tif')/10**5
    
    urban_1990 = tf.imread(urban_folder_1990 + '/urban_1990_' + str(id) + '.tif').astype(float)
    urban_1990= cv2.resize(urban_1990, (lst.shape[1], lst.shape[0]),cv2.INTER_NEAREST)
    urban_1990[urban_1990 == 0] = np.nan

    urban_2018 = tf.imread(urban_folder_2018 + '/fvc_' + str(id) + '.tif').astype(float)
    urban_2018_rsz = cv2.resize(urban_2018, (lst.shape[1], lst.shape[0]),cv2.INTER_NEAREST)
    urban_2018_rsz[urban_2018_rsz != float(id)] = np.nan
    rural_near = urban_2018_rsz * 1  # rural-urban interface
    for i in range(3*10): # 3km
        rural_near = extend_edge(rural_near)

    rural_bgr = rural_near * 1  # rural background
    for i in range(10*10): # 10km
        rural_bgr = extend_edge(rural_bgr)

    dem = tf.imread(dem_folder + '/dem_' + str(id) + '.tif').astype(float)
    dem_core = np.nanmean(dem[~np.isnan(urban_1990)])
    dem[(dem>dem_core+50) | (dem<dem_core-50)]=np.nan
    mask = np.isnan(urbanExp_frc>0.2) | np.isnan(dem)
    vegC[mask] = np.nan
    
    # Get the corresponding lst and ta_BG values
    vegC_flat = vegC.flatten()
    lst_flat = lst.flatten()
    ta_BG_flat = ta_BG.flatten()
    ndvi_flat = ndvi.flatten()
    Rn_flat = Rn.flatten()
    Q_flat = Q.flatten()

    # Find the indices of the maximum 100 values of vegC at urban, ignoring NaNs
    vegC_core = vegC * 1
    vegC_core[np.isnan(urban_1990)] = np.nan
    vegC_core_flat = vegC_core.flatten()

    valid_indices_core = np.where(~np.isnan(vegC_core_flat))[0]
    top_100_indices_core = valid_indices_core[np.argpartition(vegC_core_flat[valid_indices_core], -100)[-100:]]

    top_100_vegC_core = vegC_flat[top_100_indices_core]
    top_100_lst_core = lst_flat[top_100_indices_core][top_100_vegC_core>0.95]
    top_100_ta_BG_core = ta_BG_flat[top_100_indices_core][top_100_vegC_core>0.95]
    top_100_ndvi_core = ndvi_flat[top_100_indices_core][top_100_vegC_core>0.95]
    top_100_Rn_core = Rn_flat[top_100_indices_core][top_100_vegC_core>0.95]
    top_100_Q_core = Q_flat[top_100_indices_core][top_100_vegC_core>0.95]


    # Find the indices of the maximum 100 values of vegC at rural, ignoring NaNs
    rural_bgr[~np.isnan(rural_near)] = np.nan
    vegC_bg = vegC * 1
    vegC_bg[np.isnan(rural_bgr)] = np.nan
    vegC_bg_flat = vegC_bg.flatten()

    valid_indices_bg = np.where(~np.isnan(vegC_bg_flat))[0]
    top_100_indices_bg = valid_indices_bg[np.argpartition(vegC_bg_flat[valid_indices_bg], -100)[-100:]]

    top_100_vegC_bg = vegC_flat[top_100_indices_bg]
    top_100_lst_bg = lst_flat[top_100_indices_bg][top_100_vegC_bg>0.95]
    top_100_ta_BG_bg = ta_BG_flat[top_100_indices_bg][top_100_vegC_bg>0.95]
    top_100_ndvi_bg = ndvi_flat[top_100_indices_bg][top_100_vegC_bg>0.95]
    top_100_Rn_bg = Rn_flat[top_100_indices_bg][top_100_vegC_bg>0.95]
    top_100_Q_bg = Q_flat[top_100_indices_bg][top_100_vegC_bg>0.95]

    Tc_core = np.nanmedian(top_100_lst_core) # canopy temperature
    Ta_core = np.nanmedian(top_100_ta_BG_core) # air temperature
    ndvi_core = np.nanmean(top_100_ndvi_core) # ndvi
    Rn_core = np.nanmedian(top_100_Rn_core)
    Q_core = np.nanmedian(top_100_Q_core)


    Tc_bg = np.nanmedian(top_100_lst_bg)  # canopy temperature
    Ta_bg = np.nanmedian(top_100_ta_BG_bg)  # air temperature
    ndvi_bg = np.nanmean(top_100_ndvi_bg) # ndvi
    Rn_bg = np.nanmedian(top_100_Rn_bg)
    Q_bg = np.nanmedian(top_100_Q_bg)

    uhi = np.nanmean(ta_BG[~np.isnan(vegC_core)]) - np.nanmean(ta_BG[~np.isnan(vegC_bg)])
    print(id)

    ws_i = ws[ws['ID'] == id]['mean'] / 100
    Td_i = Td[Td['ID'] == id]['MATd']

    if abs(Tc_bg-Tc_core)>12: Tc_core=Ta_core+5; Tc_bg = Ta_bg+5
    # Modify these lines to extract scalar values
    H_core = cal_H(Td_i, Tc_core, Ta_core, ws_i).values[0]
    H_bg = cal_H(Td_i, Tc_bg, Ta_bg, ws_i).values[0]

    return [id, Tc_core, Ta_core, ndvi_core, Rn_core, Q_core, H_core, 
            Tc_bg, Ta_bg, ndvi_bg, Rn_bg, Q_bg, H_bg, uhi]

## main ##
lst_dir_path = os.path.join('..', '..', 'project 1 urban vegetation thermoregulation', '1_Input', 'LST')

treeC_folder = os.path.join('..', '..', 'project 1 urban vegetation thermoregulation', '1_Input', 'treeCover_100m')

grassC_folder = os.path.join('..', '..', 'project 1 urban vegetation thermoregulation', '1_Input', 'grassCover_100m')

urbanExp_folder = os.path.join( '..', '1_Input', 'urban_expansion_frac_500m')

ta_MDS_folder = os.path.join('..','..','urban_env_data','TaMax_modis_monthly1000m')

ta_ERA_folder = os.path.join('..','1_Input','ta_anom')
ta_filenames = os.listdir(ta_ERA_folder)

ndvi_folder = os.path.join('..','..','project 1 urban vegetation thermoregulation','1_Input','NDVI')

urban_folder_1990 = os.path.join('..','1_Input','urban','urban_1990_250m')
urban_folder_2018 = os.path.join('..','1_Input','urban','urban_2018_1000m')

dem_folder = os.path.join('..','..','project 1 urban vegetation thermoregulation','1_Input','DEM_100m')

sw_folder = os.path.join('..','..','urban_env_data','shortwave')
lw_folder = os.path.join('..','..','urban_env_data','longwave')
alb_folder = os.path.join('..','..','urban_env_data','Albedo')
ratios = pd.read_csv(os.path.join('..','..','urban_env_data','noontime_energy_ratio.csv'))
ahe_folder = os.path.join('..','..','project 1 urban vegetation thermoregulation','1_Input','AHE_100m')
ws = pd.read_csv(os.path.join('..','..','urban_env_data','ws_csv_urban_751.csv'))
Td = pd.read_csv(os.path.join('..','..','urban_env_data','MATd_urban_751.csv'))
current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate = pd.merge(climate,Td,on='ID')

# Calculate dT
dT = climate['MAT'].mean() - climate['MATd'].mean()
climate['MATd'] = climate['MATd'].fillna(climate['MAT'] - dT)
Td = climate[['ID','MATd']];

IDs = []
for name in ta_filenames:
    id = float(name.split('_')[-1].split('.npy')[0])
    IDs.append(id)
IDs = np.sort(IDs)

# Use parallel processing to speed up the loop
results = Parallel(n_jobs=-1)(delayed(process_id)(id) for id in IDs)

# Filter out None results
results = [result for result in results if result is not None]

results_arr = np.array(results)
# Convert results_arr to DataFrame
columns = ['ID', 'Tc_core', 'Ta_core', 'ndvi_core', 'Rn_core', 'Q_core', 'H_core', 'Tc_bg', 'Ta_bg', 'ndvi_bg','Rn_bg', 'Q_bg', 'H_bg', 'uhi']
results_df = pd.DataFrame(results_arr, columns=columns)

# Print the DataFrame
print(results_df)
output_dir = os.path.join('..', '2_Output', 'drivers', 'urban_factors_effect_v2.csv')
results_df.to_csv(output_dir, index=False)