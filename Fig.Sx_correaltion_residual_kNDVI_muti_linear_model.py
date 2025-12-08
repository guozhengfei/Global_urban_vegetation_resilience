import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.signal as ss
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")
import cv2
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

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
    # urban_folder = current_dir+'/1_Input/urban_area/'
    filenames = os.listdir(current_dir + '/1_Input/ta_anom/')

    IDs = []
    for name in filenames:
        id = float(name.split('_')[-1].split('.npy')[0])
        IDs.append(id)

    IDs = np.sort(IDs)
    IDs_2 = np.load(current_dir+'/2_Output/tac_nadir_city_3zones.npz')['array2']
    IDs_diff = set(IDs) - set(IDs_2)

    corrs = []
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
        # plt.figure(); plt.plot(np.linspace(2001,2021+11/12,264),EVI[1,:])
        # plt.figure(); plt.plot(np.linspace(2001,2021+11/12,264),EVI_sg.T[1,:])

        EVI = EVI_rsp[~crop_mask_1d, :]
        EVI = IQR_filter2(EVI)
        
        # Fill NaN values using 5-year climatology
        EVI = fill_nan_with_climatology(EVI)
        
        # EVI_sg = ss.savgol_filter(EVI.T, 4, 3, mode='nearest', axis=0)
        # EVI_sg[EVI_sg < 0] = 0
        # ser = EVI_sg.T
        ser = EVI
        if ser.shape[0]<10: continue
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

        variables_all = np.stack((res_ndvi[:, :-1], res_ta[:, 1:], res_rad[:, 1:], res_pr[:, 1:], res_ndvi[:, 1:]), axis=2)

        variables_all = variables_all[~mask, :, :]
        array = np.nanmean(variables_all,axis=0)
        X = array[:, :-1]
        y = array[:, -1]
        regressor = LinearRegression()
        regressor.fit(X, y)
        y_pred = regressor.predict(X)

        # Calculate Pearson correlation and RMSE
        corr, p_value = pearsonr(y, y_pred)
        rmse = np.sqrt(np.nanmean((y - y_pred) ** 2))

        # store id, p-value and RMSE
        corrs.append([id, p_value, rmse])
        print(id)
    corrs_df = pd.DataFrame(np.array(corrs))
    corrs_df.columns = ['id', 'p_value', 'rmse']

    # save stats table
    output_file = current_dir + '/2_Output/' + 'Mutilinear_model_stats.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    corrs_df.to_csv(output_file, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.5))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot p-value histogram (focus on small p-values)
    p_values = corrs_df['p_value'].values
    p_values[p_values>=0.05]=np.nan
    p_values = p_values[~np.isnan(p_values)]
    # ax1.hist(p_values, bins=30, range=(0, 0.05),
    #          color='#4e79a7', edgecolor='white', alpha=0.85)
    counts, bins = np.histogram(p_values, bins=40)
    counts_perc = counts / counts.sum() * 100.0
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    ax1.bar(bin_centers, counts_perc, width=(bins[1] - bins[0]) * 0.95,
            color='#4e79a7', edgecolor='k', alpha=0.85)

    # Annotate mean p-value (within plotted range)
    p_mask = p_values <= 0.05
    if np.any(p_mask):
        mean_p = np.mean(p_values[p_mask])
        ax1.axvline(mean_p, color='#e15759', linestyle='--', linewidth=1.5)
        ax1.text(mean_p + 0.002, ax1.get_ylim()[1] * 0.9,
                 f'Mean: {mean_p:.3f}', color='#e15759', fontsize=10)

    # Format p-value plot
    ax1.set_title('P-value Distribution', pad=10, fontsize=12, fontweight='semibold')
    ax1.set_xlabel('P-value', fontsize=10)
    ax1.set_ylabel('Density (%)', fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Plot RMSE histogram
    rmses = corrs_df['rmse'].values
    # ax2.hist(rmses, bins=30, color='#59a14f', edgecolor='white', alpha=0.85)
    counts, bins = np.histogram(rmses, bins=3)
    counts_perc = counts / counts.sum() * 100.0
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    ax2.bar(bin_centers, counts_perc, width=(bins[1] - bins[0]) * 0.95,
           color='#59a14f', edgecolor='k', alpha=0.85)

    # Annotate mean RMSE
    mean_rmse = np.nanmean(rmses)
    ax2.axvline(mean_rmse, color='#e15759', linestyle='--', linewidth=1.5)
    ax2.text(mean_rmse + 0.02 * (ax2.get_xlim()[1]-ax2.get_xlim()[0]), ax2.get_ylim()[1] * 0.9,
             f'Mean: {mean_rmse:.3f}', color='#e15759', fontsize=10)

    # Format RMSE plot
    ax2.set_xlim([0,0.9])
    ax2.set_title('RMSE Distribution', pad=10, fontsize=12, fontweight='semibold')
    ax2.set_xlabel('RMSE', fontsize=10)
    ax2.set_ylabel('Density (%)', fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # Adjust layout and save figure
    plt.tight_layout()
    figToPath = current_dir + '/4_Figures/FigS_multiple_linear_results.png'
    os.makedirs(os.path.dirname(figToPath), exist_ok=True)
    fig.savefig(figToPath, dpi=600, bbox_inches='tight')
    print(f'Figure saved to: {figToPath}')
