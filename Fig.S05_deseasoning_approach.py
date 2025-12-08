import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib;
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import scipy.signal as ss
import random

if __name__=='__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

    evi = [0.01994839, 0.02373375, 0.0819062,0.1994839, 0.27881973, 0.3594839, 0.38181027, 0.38584479, 0.26493029, 0.14740829, 0.05424056, 0.02278374]

    evi_40yr = np.repeat(evi, 30, axis=0).reshape(12, 30)
    evi_40yr = evi_40yr.T.reshape(-1)
    random_numbers = [random.uniform(0, 0.025) for _ in range(12*30)]

    # linear greening
    devis = []
    for i in range(30):
        devi = (np.array(evi) - 0.01994839) * (0.3 * i / 30)
        devis.append(devi)
    devis = np.array(devis)

    ndvi = evi_40yr + devis.reshape(-1) + random_numbers

    # remove long-term mean
    ndvi_reshaped = ndvi[:,np.newaxis].T
    ser = ndvi_reshaped * 1
    yr_num = 30
    bands_year = 12
    EVI_yr = np.zeros_like(ser)
    for year in range(yr_num):
        st = year * bands_year
        ed = st + bands_year
        evi_year = np.nanmean(ser[:, st:ed], axis=1)
        evi_year_rep = np.repeat(evi_year[:, np.newaxis], bands_year, axis=1)
        EVI_yr[:, st:ed] = evi_year_rep
    rm_offline = ser - EVI_yr

    # remove seasonality
    Evi_sea_rep = np.zeros_like(rm_offline)
    for yr in range(yr_num):
        start_index = (yr - 3) * bands_year
        if start_index < 0: start_index = 0
        end_index = (yr + 4) * bands_year
        if end_index > bands_year * yr_num: end_index = bands_year * yr_num
        data_i = rm_offline[:, start_index:end_index]
        Evi_sea = np.mean(np.reshape(data_i, (data_i.shape[0], int(data_i.shape[1] / bands_year), bands_year)),
                          axis=1)
        Evi_sea_rep[:, yr * bands_year:(yr + 1) * bands_year] = Evi_sea

    res = rm_offline - Evi_sea_rep

    # remove seasonality static
    Evi_sea_rep2 = np.zeros_like(rm_offline)
    seaonality = np.nanmean(np.reshape(rm_offline,[int(rm_offline.shape[1]/bands_year),bands_year]),axis=0)
    for yr in range(yr_num):
        Evi_sea_rep2[:, yr * bands_year:(yr + 1) * bands_year] = seaonality

    res2 = rm_offline - Evi_sea_rep2

    # Create a 3x2 subplot
    fig, axs = plt.subplots(4, 2, figsize=(10, 12))

    time_axis = np.linspace(1991, 2020 + 11 / 12, 360)

    # Plot 1: Smoothed NDVI
    axs[0, 0].plot(time_axis, evi_40yr + devis.reshape(-1))
    axs[0, 0].set_title('Smoothed NDVI')

    # Plot 2: noise
    axs[0, 1].plot(time_axis, random_numbers - np.mean(random_numbers))
    axs[0, 1].set_title('Random noises')
    axs[0, 1].set_ylim([-0.03,0.03])

    # Plot 3: Original NDVI
    axs[1, 0].plot(time_axis, ndvi)
    axs[1, 0].set_title('Original NDVI')

    # Plot 2: NDVI with long-term mean removed
    axs[1, 1].plot(time_axis, rm_offline[0, :])
    axs[1, 1].set_title('Detrend NDVI')

    # Plot 3: Moving window seasonality
    axs[2, 0].plot(time_axis, Evi_sea_rep[0, :])
    axs[2, 0].set_title('Rolling dynamic Seasonality')

    # Plot 4: Residuals after moving window seasonality removal
    axs[3, 0].plot(time_axis, res[0, :])
    axs[3, 0].set_title('Residual NDVI')
    axs[3, 0].set_ylim([-0.03, 0.03])

    # Plot 5: Static seasonality
    axs[2, 1].plot(time_axis, Evi_sea_rep2[0, :])
    axs[2, 1].set_title('Static Seasonality')

    # Plot 6: Residuals after static seasonality removal
    axs[3, 1].plot(time_axis, res2[0, :])
    axs[3, 1].set_title('Residual NDVI')
    axs[3, 1].set_ylim([-0.03, 0.03])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8, wspace=0.4)  # 增大垂直和水平间距
    plt.show()

    figToPath = current_dir + '/4_Figures/FigS05_deseason_approach'
    fig.savefig(figToPath, dpi=900, bbox_inches='tight')