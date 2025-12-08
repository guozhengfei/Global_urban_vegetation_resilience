import numpy as np
import tifffile as tf
import matplotlib;

from additional_exploration.step_06_cal_TAC_opt_variance_copy import IQR_filter2

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
plt.close()

import os
import warnings
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
warnings.filterwarnings("ignore")
import scipy.stats as st

def IQR_filter(array):
    p25 = np.nanpercentile(array, 25)
    p75 = np.nanpercentile(array,75)
    IQR = p75-p25
    maxV = p75 + 1.5*IQR
    minV = p25 - 1.5 * IQR
    array[array < minV] = np.nan
    array[array > maxV] = np.nan
    arraynew = array
    return arraynew
if __name__ == '__main__':
    # Identify the urban_rural tac diff, and sort it
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    ID = np.sort(np.load(current_dir + '/2_Output/tac_landsat_city_3zones.npz')['array2'])[1:]

    ndvi_folder1 = current_dir + '/2_Output/VI_Landsat/'
    ndvi_folder2 = current_dir + '/1_Input/nadir_ndvi_15d_500/'

    # Paths to data folders
    tac_folder1 = current_dir + '/2_Output/tac_Landsat/'
    tac_folder2 = current_dir + '/2_Output/tac_500m_yr/'

    R = []

    for id in ID:  # Iterate over each subplot
        try:
            tac1_i = np.load(tac_folder1 + 'tac_' + str(id) + '.npy')
            tac1_spa_mean = IQR_filter(np.nanmean(tac1_i, axis=1))

            tac2_i = np.load(tac_folder2 + 'tac_' + str(id) + '.npy')
            tac2_spa_mean = IQR_filter(np.nanmean(tac2_i, axis=2))

            var1_i = np.load(tac_folder1 + 'tac_' + str(id) + '_variance.npy')
            var1_spa_mean = IQR_filter(np.nanmean(var1_i, axis=1))

            var2_i = np.load(tac_folder2 + 'tac_' + str(id) + '_variance.npy')
            var2_spa_mean = IQR_filter(np.nanmean(var2_i, axis=2))

            ndvi1 = pd.read_csv(ndvi_folder1 + 'NDVI_8d_'+str(id)+'.csv').iloc[:,:-2].astype(float)
            ndvi1_mean = np.nanmean(ndvi1,axis=1)

            ndvi2 = tf.imread(ndvi_folder2 + 'nadir_ndvi_15d_' + str(id) + '.tif')
            ndvi2_mean = np.nanmean(ndvi2, axis=2)

            mask1 = (ndvi1_mean<0) | np.isnan(tac1_spa_mean) | np.isnan(var1_spa_mean)
            ndvi1_clean = ndvi1_mean[~mask1]
            tac1_spa_mean_clean = tac1_spa_mean[~mask1]
            var1_spa_mean_clean = var1_spa_mean[~mask1]

            mask2 = (ndvi2_mean < 0) | np.isnan(tac2_spa_mean) | np.isnan(var2_spa_mean)
            ndvi2_clean = ndvi2_mean[~mask2]
            tac2_spa_mean_clean = tac2_spa_mean[~mask2]
            var2_spa_mean_clean = var2_spa_mean[~mask2]

            # Landsat
            r1 = np.corrcoef(ndvi1_clean, tac1_spa_mean_clean)[0,1]
            r1_var = np.corrcoef(ndvi1_clean, var1_spa_mean_clean)[0,1]

            # MODIS
            r2 = np.corrcoef(ndvi2_clean, tac2_spa_mean_clean)[0, 1]
            r2_var = np.corrcoef(ndvi2_clean, var2_spa_mean_clean)[0, 1]

            R.append([id, r1, r1_var, r2, r2_var])
            print(id)
        except FileNotFoundError: continue

    # Prepare array of correlations
    R_arr = np.array(R)
    if R_arr.size == 0:
        raise SystemExit("No correlation results found; check input files.")

    # Absolute correlations for plotting
    landsat_tac = np.abs(R_arr[:, 1].astype(float))
    landsat_var = np.abs(R_arr[:, 2].astype(float))
    modis_tac = np.abs(R_arr[:, 3].astype(float))
    modis_var = np.abs(R_arr[:, 4].astype(float))

    # Plotting: two subplots (Landsat left, MODIS right) with histograms + KDE lines
    def hist_with_kde(ax, data_hist, data_var, title, colors=('C0','C1')):
        # Histogram (density)
        h_vals, h_bins, _ = ax.hist(data_hist, bins=bins, density=True, color=colors[0], alpha=alpha, label='TAC (abs)', edgecolor='none')
        hv_vals, hv_bins, _ = ax.hist(data_var, bins=bins, density=True, color=colors[1], alpha=alpha, label='Variance (abs)', edgecolor='none')
        # KDE lines
        x_grid = np.linspace(0, 1, 1000)
        try:
            kde1 = st.gaussian_kde(data_hist[~np.isnan(data_hist)])
            ax.plot(x_grid, kde1(x_grid), color=colors[0], lw=1.6)
        except Exception:
            pass
        try:
            kde2 = st.gaussian_kde(data_var[~np.isnan(data_var)])
            ax.plot(x_grid, kde2(x_grid), color=colors[1], lw=1.6)
        except Exception:
            pass
        # Mean lines and annotations
        m1 = np.nanmean(data_hist); m2 = np.nanmean(data_var)
        ax.axvline(m1, color=colors[0], linestyle='--', lw=1)
        ax.axvline(m2, color=colors[1], linestyle='--', lw=1)
        ax.text(0.98, 0.95, f'Mean TAC={m1:.2f}\nMean Var={m2:.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=11, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        ax.set_xlim(0, 1)
        ax.set_xlabel('Absolute correlation', fontsize=12)
        # ax.set_title(title, fontsize=13)
        ax.legend(frameon=False, fontsize=11)


    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(12 * 0.7, 10 * 0.7))
    bins = 20
    alpha = 0.6

    hist_with_kde(axes[1,0], landsat_tac, landsat_var, 'Landsat')
    hist_with_kde(axes[1,1], modis_tac, modis_var, 'MODIS')

    axes[1,0].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)

    axes[0,0].plot(ndvi2_clean, tac2_spa_mean_clean,'o',alpha=0.1)
    z = np.polyfit(ndvi2_clean, tac2_spa_mean_clean, 1)
    p = np.poly1d(z)
    axes[0,0].plot(np.unique(ndvi2_clean), p(np.unique(ndvi2_clean)),
            'r-', linewidth=1.5)
    axes[0, 1].plot(ndvi2_clean, var2_spa_mean_clean, 'o', alpha=0.1)
    z = np.polyfit(ndvi2_clean, var2_spa_mean_clean, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(np.unique(ndvi2_clean), p(np.unique(ndvi2_clean)),
                    'r-', linewidth=1.5)
    axes[0, 0].set_ylabel('TAC', fontsize=12)
    axes[0, 1].set_ylabel('Variance', fontsize=12)
    axes[0, 0].set_xlim([0.12,0.39])
    axes[0, 1].set_xlim([0.12, 0.39])
    axes[0, 1].set_ylim([-0.0003, 0.0016])

    axes[0, 0].set_xlabel('kNDVI', fontsize=12)
    axes[0, 1].set_xlabel('kNDVI', fontsize=12)
    plt.tight_layout()

    # Optionally save
    out_png = os.path.join(current_dir, '4_Figures', 'fig_s01g_tac_variance_ndvi_examples.png')

    fig.savefig(out_png, dpi=600, bbox_inches='tight')

