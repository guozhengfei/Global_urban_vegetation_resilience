import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib;

#matplotlib.use('Qt5Agg')
import scipy.signal as ss
import multiprocess as mp
import os
import warnings
import cv2
warnings.filterwarnings("ignore")
import numpy as np
import scipy.stats as st
import pandas as pd

if __name__ == '__main__':
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    relative_path = '/2_Output/tac_nadir_city_3zones_modis_month.npz'
    
    # Load TAC data and normalize by subtracting zone-specific means
    df_tac = np.load(current_dir + relative_path)['array1']
    ID = np.sort(np.load(current_dir + relative_path)['array2'])
    
    # Subtract zone-specific means for each city
    for i in range(df_tac.shape[0]):  # Loop through cities
        for zone in range(3):  # Loop through zones (0: urban, 1: interface, 2: rural)
            zone_mean = np.nanmean(df_tac[i, zone, :])  # Calculate mean for specific zone
            df_tac[i, zone, :] = df_tac[i, zone, :] - zone_mean

    relative_path2 = '/2_Output/tac_landsat_city_3zones.npz'
    
    # Load and filter PDSI data
    df_pdsi = pd.read_csv(current_dir + '/2_Output/pdsi_timeseries_all.csv')
    df_pdsi = df_pdsi.groupby('ID').mean().reset_index()
    df_pdsi = df_pdsi[df_pdsi['ID'].isin(ID)].sort_values('ID').reset_index(drop=True)
    pdsi = df_pdsi.iloc[:,1:-3].values
    
    # Load and filter TMMX data
    df_tmmx = pd.read_csv(current_dir + '/2_Output/tmmx_timeseries_all.csv')
    df_tmmx = df_tmmx.groupby('ID').mean().reset_index()
    df_tmmx = df_tmmx[df_tmmx['ID'].isin(ID)].sort_values('ID').reset_index(drop=True)
    tmmx = df_tmmx.iloc[:,1:-3].values
    
    # Initialize arrays to store mean TACs during extreme conditions
    n_cities = len(ID)
    tac_during_drought = np.zeros((n_cities, 3))
    tac_during_heat = np.zeros((n_cities, 3))
    
    # Calculate temperature anomalies and threshold
    tmmx_mean = np.mean(tmmx, axis=1, keepdims=True)
    tmmx_anomaly = tmmx - tmmx_mean
    
    # Loop through each city
    for i in range(n_cities):
        # Find drought months (PDSI < -300)
        drought_months = np.where(pdsi[i, :] < -400)[0]+1
        drought_months[drought_months>262]=262
        
        # Find hot months (tmmx > 200 and anomaly > 90th percentile)
        anomaly_threshold = np.nanpercentile(tmmx_anomaly[i, :], 90)
        hot_months = np.where((tmmx[i, :] > 250) &
                            (tmmx_anomaly[i, :] > anomaly_threshold))[0]+1
        hot_months[hot_months > 262] = 262
        
        # Calculate TACs during drought
        if len(drought_months) > 0:
            drought_tacs = df_tac[i, :, drought_months]
            tac_during_drought[i, :] = np.nanmean(drought_tacs, axis=0)
        else:
            tac_during_drought[i, :] = np.nan
            
        # Calculate TACs during heat
        if len(hot_months) > 0:
            heat_tacs = df_tac[i, :, hot_months]
            tac_during_heat[i, :] = np.nanmean(heat_tacs, axis=0)
        else:
            tac_during_heat[i, :] = np.nan
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'ID': ID,
        'Urban_TAC_drought': tac_during_drought[:, 0],
        'Interface_TAC_drought': tac_during_drought[:, 1],
        'Rural_TAC_drought': tac_during_drought[:, 2],
        'Urban_TAC_heat': tac_during_heat[:, 0],
        'Interface_TAC_heat': tac_during_heat[:, 1],
        'Rural_TAC_heat': tac_during_heat[:, 2]
    })
    
    # Calculate and print summary statistics
    print("\nMean TAC during extreme conditions:")
    print("\nDrought periods (PDSI < -300):")
    print(results[['Urban_TAC_drought', 'Interface_TAC_drought', 'Rural_TAC_drought']].describe())
    print("\nHeat periods (TMMX > 200 & anomaly > 90th percentile):")
    print(results[['Urban_TAC_heat', 'Interface_TAC_heat', 'Rural_TAC_heat']].describe())
    
    # Save results
    output_path = os.path.join(current_dir, '2_Output', 'tac_during_extremes.csv')
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")




