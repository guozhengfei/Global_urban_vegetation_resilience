import numpy as np
import tifffile as tf
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.signal as ss
import multiprocess as mp
import os
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
warnings.filterwarnings("ignore")
import cv2
import pandas as pd

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

def check_vif(array):
    """
    Calculates VIF for a pandas DataFrame of predictors.
    """
    df_predictors = pd.DataFrame(array)
    # 1. Drop NaNs (VIF cannot handle missing values)
    df_clean = df_predictors.dropna()
    X = add_constant(df_clean)

    # 3. Calculate VIF for each predictor
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(len(X.columns))]

    return vif_data["VIF"].values[1:]
if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    EVI_folder = current_dir + '/1_Input/nadir_ndvi_15d_500/'
    grass_folder = current_dir + '/1_Input/grassCover_250m/'
    tree_folder = current_dir + '/1_Input/treeCover_250m/'
    crop_folder = current_dir + '/1_Input/cropCover_250m/'
    water_folder = current_dir + '/1_Input/waterCover_250m/'
    urbanExp_folder = current_dir + '/1_Input/urban_expansion_frac_500m/'
    # urban_folder = current_dir+'/1_Input/urban_area/'
    filenames = os.listdir(current_dir + '/1_Input/ta_anom/')

    IDs = []
    for name in filenames:
        id = float(name.split('_')[-1].split('.npy')[0])
        IDs.append(id)

    IDs = np.sort(IDs)
    IDs_2 = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array2']

    VIFs = []
    for id in IDs:
        EVI0 = tf.imread(EVI_folder + 'nadir_ndvi_15d_' + str(id) + '.tif')[:, :, :264*2][:,:,::2]
        nan_frac = np.sum(np.isnan(EVI0),axis=2)/EVI0.shape[2]
        # plt.figure(); plt.hist(nan_frac.reshape(-1),50)
        crop_frc = tf.imread(crop_folder + 'cropC_' + str(id) + '.tif')
        crop_frc = cv2.resize(crop_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)
        # plt.figure(); plt.imshow(water_frc)

        grass_frc = tf.imread(grass_folder + 'grassC_' + str(id) + '.tif')
        grass_frc = cv2.resize(grass_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)

        tree_frc = tf.imread(tree_folder + 'treeC_' + str(id) + '.tif')
        tree_frc = cv2.resize(tree_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)

        water_frc = tf.imread(water_folder + 'waterC_' + str(id) + '.tif')
        water_frc = cv2.resize(water_frc, (EVI0.shape[1], EVI0.shape[0]), cv2.INTER_LINEAR)

        urbanExp_frc = tf.imread(urbanExp_folder + 'urban_exp_C_' + str(id) + '.tif')

        # remove the pixel with crop > 20% or urban expansion or miss data >30%
        crop_frc[crop_frc > 0.2] = np.nan
        veg_nature = grass_frc + tree_frc
        crop_frc[crop_frc > veg_nature] = np.nan
        mask_miss = np.sum(np.isnan(EVI0), axis=2) / EVI0.shape[2]
        mask = np.isnan(crop_frc) | (mask_miss > 0.3) | (water_frc>0.3) | (urbanExp_frc>0.2)
        # plt.figure(); plt.imshow(mask)
        crop_mask_1d = mask.flatten()
        EVI_rsp = EVI0.reshape((EVI0.shape[0] * EVI0.shape[1], EVI0.shape[2]))

        EVI = EVI_rsp[~crop_mask_1d, :]
        EVI = IQR_filter2(EVI)
        
        # Fill NaN values using 5-year climatology
        EVI = fill_nan_with_climatology(EVI)
        
        # EVI_sg = ss.savgol_filter(EVI.T, 4, 3, mode='nearest', axis=0)
        # EVI_sg[EVI_sg < 0] = 0
        # ser = EVI_sg.T
        ser = EVI
        # if ser.shape[0]<10: continue
        EVI_yr = np.zeros_like(ser)

        yr_num = 22
        bands_year = 12
        for year in range(yr_num):
            st = year * bands_year
            ed = st + bands_year
            evi_year = np.nanmean(ser[:, st:ed], axis=1)
            evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
            EVI_yr[:, st:ed] = evi_year_rep
        rm_offline = ser - EVI_yr
        # plt.figure(); plt.plot(np.nanmean(rm_offline, axis=0))

        del EVI_yr, ser

        Evi_sea_rep = np.zeros_like(rm_offline)
        for yr in range(yr_num):
            start_index = (yr - 4) * bands_year
            if start_index < 0: start_index = 0
            end_index = (yr + 5) * bands_year
            if end_index > bands_year * yr_num: end_index = bands_year * yr_num
            data_i = rm_offline[:, start_index:end_index]
            Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)),
                              axis=1)
            Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea

        # res = rm_offline - Evi_sea_rep
        res_ndvi = rm_offline - Evi_sea_rep
        # plt.figure(); plt.plot(np.nanmean(res_pr, axis=0))
        res_ndvi[np.isnan(res_ndvi)]=0

        nan_frac_ndvi = np.sum(np.isnan(res_ndvi), axis=1) / res_ndvi.shape[1]
        res_pr_org = np.load(current_dir + '/1_Input/pre_anom/pr_month_anomaly_'+str(id)+'.npy')[:, :,1:1+264]
        res_pr = res_pr_org.reshape((res_pr_org.shape[0]*res_pr_org.shape[1],res_pr_org.shape[2]))[~crop_mask_1d, :]
        nan_frac_pr = np.sum(np.isnan(res_pr), axis=1) / res_pr.shape[1]

        res_rad_org = np.load(current_dir + '/1_Input/rad_anom/rad_month_anomaly_' + str(id) + '.npy')[:, :, 1:1 + 264]
        res_rad = res_rad_org.reshape((res_rad_org.shape[0] * res_rad_org.shape[1], res_rad_org.shape[2]))[~crop_mask_1d, :]
        nan_frac_rad = np.sum(np.isnan(res_rad), axis=1) / res_rad.shape[1]

        res_ta_org = np.load(current_dir + '/1_Input/ta_anom/ta_month_anomaly_' + str(id) + '.npy')[:, :, 1:1 + 264]
        res_ta = res_ta_org.reshape((res_ta_org.shape[0] * res_ta_org.shape[1], res_ta_org.shape[2]))[~crop_mask_1d, :]
        nan_frac_ta = np.sum(np.isnan(res_ta), axis=1) / res_ta.shape[1]

        mask = (nan_frac_ndvi > 0.75) | (nan_frac_pr > 0.3) | (nan_frac_rad > 0.3) | (nan_frac_ta > 0.3)

        variables_all = np.stack((res_ta[:, 1:], res_rad[:, 1:], res_pr[:, 1:], res_ndvi[:, 1:]), axis=2)
        variables_all = variables_all[~mask, :, :]
        del rm_offline, Evi_sea_rep
        variables_all = variables_all[::10,:,:]
        divided_arrays = [row for row in variables_all]

        with mp.Pool(10) as pool:
            results = list(pool.map(check_vif, divided_arrays))
        vif_arr = np.array(results)
        VIFs.append(vif_arr)
        print(id)

    VIFs_2 = []
    for ele in VIFs:
        if ele.shape[0] > 1:
            VIFs_2.append(ele)

    # combine all VIF arrays into one 2D array (rows = samples, cols = variables)
    if len(VIFs_2) == 0:
        raise RuntimeError("No valid VIF results to plot.")
    VIFs_arr = np.concatenate(VIFs_2, axis=0)

    # --- Plot: three histograms as subplots, y-axis in percent (0-100%) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(15*0.5, 5*0.5), sharey=True)

    labels = ['Air temperature', 'Radiation', 'Precipitation']
    colors = ['#2b8cbe', '#7bccc4', '#fdae61']

    for i, ax in enumerate(axes):
        data = VIFs_arr[:, i]
        data = data[~np.isnan(data)]
        if data.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            continue

        counts, bins = np.histogram(data, bins=20)
        counts_perc = counts / counts.sum() * 100.0
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        ax.bar(bin_centers, counts_perc, width=(bins[1] - bins[0]) * 0.95,
               color=colors[i], edgecolor='k', alpha=0.85)

        ax.set_xlim(0, 5)
        ax.set_ylim(0, 40)  # 0 - 100% as requested
        ax.set_title(f'{labels[i]} (VIF)', fontsize=12, fontweight='semibold')
        ax.set_xlabel('VIF', fontsize=11)
        if i == 0:
            ax.set_ylabel('Density (%)', fontsize=11)

        # median and mean annotations
        median_val = np.nanmedian(data)
        mean_val = np.nanmean(data)
        ax.axvline(median_val, color='k', linestyle='--', linewidth=1)
        ax.axvline(mean_val, color='gray', linestyle=':', linewidth=1)
        ax.text(0.98, 0.95, f'Mean={mean_val:.2f}\nN={"{:.2e}".format(data.size*12)}',
                ha='right', va='top', transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # save and show
    out_dir = os.path.join(current_dir, '4_Figures')
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, 'vif_histograms.png')
    fig.savefig(fig_path, dpi=600, bbox_inches='tight')
    print(f'VIF histograms saved to: {fig_path}')
    plt.show()


