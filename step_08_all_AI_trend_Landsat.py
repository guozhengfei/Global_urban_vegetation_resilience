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
    current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
    folder = current_dir + '/urban_env_data/NTL_gdp_500m/'
    AI = pd.read_csv(current_dir + '/2_Output/AI_2000-2022.csv')

    IDs = set(AI['ID'])

    trends = []
    for id in IDs:
        AI_i = AI[AI['ID'] == id]['AI'].values
        PR_i = AI[AI['ID'] == id]['PR'].values
        PET_i = AI[AI['ID'] == id]['PET'].values

        trends.append([id, cal_slope(AI_i),cal_slope(PR_i),cal_slope(PET_i)])
        print(id)


    trends_arr = np.array(trends)
    import pandas as pd
    df_trends = pd.DataFrame(trends_arr,
                             columns=['ID','AI_trend','PR_trend','PET_trend'])

    output_csv = os.path.join(current_dir, '2_Output', 'AI_trends.csv')
    df_trends.to_csv(output_csv, index=False)

    # tac_trend
    df_tac = pd.read_csv(current_dir + '/2_Output/tac_trends.csv')

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
    plt.savefig(os.path.join(current_dir, '4_Figures', 'AI_trend_correlations_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
