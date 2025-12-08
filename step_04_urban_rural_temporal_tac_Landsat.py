import numpy as np
import tifffile as tf
import matplotlib;
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.signal as ss
import multiprocess as mp
import os
import warnings
import cv2
warnings.filterwarnings("ignore")
import numpy as np

def calculate_yearly_mean(monthly_data):
    """Convert monthly data to yearly by taking mean of each 12 months"""
    return monthly_data.reshape(-1, 12).mean(axis=1)

def calculate_linear_trend(y):
    """Calculate linear trend slope handling NaN values"""
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
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    output_file = current_dir + '/2_Output/' + 'tac_landsat_city_3zones' + '.npz'
    TAC_arr = np.load(output_file)['array1']
    IDs = np.load(output_file)['array2']
    
    # Reshape monthly data to yearly (750 cities × 3 regions × 24 years)
    TAC_arr_yearly = np.zeros((750, 3, 24))
    for city in range(750):
        for region in range(3):
            TAC_arr_yearly[city, region] = calculate_yearly_mean(TAC_arr[city, region])
    
    # Calculate linear trends (750 cities × 3 regions)
    TAC_trends = np.zeros((750, 3))
    for city in range(750):
        for region in range(3):
            TAC_trends[city, region] = calculate_linear_trend(TAC_arr_yearly[city, region])
    
    plt.figure(); plt.plot(TAC_trends[:,0],TAC_trends[:,-1],'o')
    # Save results
    mask = np.isnan(TAC_trends[:,0]) | np.isnan(TAC_trends[:,-1])
    import scipy.stats as st
    st.linregress(TAC_trends[:,0][~mask],TAC_trends[:,-1][~mask])
    st.linregress(TAC_trends[:, 0][~mask],TAC_trends[:, 0][~mask]- TAC_trends[:, -1][~mask])
    plt.figure(); plt.plot(TAC_trends[:, 0],TAC_trends[:, 0]-TAC_trends[:, -1], 'o')
    
    # Convert TAC_trends and IDs to DataFrame
    import pandas as pd
    
    df_trends = pd.DataFrame(TAC_trends, 
                           columns=['urban_core_trend', 'urban_edge_trend', 'rural_trend'])
    df_trends['ID'] = IDs
    
    # Calculate trend differences
    df_trends['urban_rural_diff'] = df_trends['urban_core_trend'] - df_trends['rural_trend']
    
    # Save to CSV
    output_csv = os.path.join(current_dir, '2_Output', 'tac_trends.csv')
    df_trends.to_csv(output_csv, index=False)
    print(f"Trends saved to: {output_csv}")

