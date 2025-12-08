import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
    folder = current_dir + "/1_Input/urban_flux/Urban-PLUMBER_FullCollection_v1/"
    subfolders = [x for x in os.listdir(folder) if not x.startswith('.')]

    # List to collect all monthly dataframes
    all_monthly = []

    for f in subfolders:
        ds1 = xr.open_dataset(folder+ f +'/timeseries/'+f+'_clean_observations_v1.nc')
        df1 = ds1.to_dataframe().reset_index()
        df1 = df1[[col for col in df1.columns if '_qc' not in col]]

        ds2 = xr.open_dataset(folder + f + '/timeseries/' + f + '_metforcing_v1.nc')
        df2 = ds2.to_dataframe().reset_index()
        # Keep only rows in df2 with the same time as df1
        df2 = df2[df2['time'].isin(df1['time'])]
        # Replace df1 columns with df2 data for columns present in both
        common_cols = [col for col in df1.columns if col in df2.columns and col not in ['time', 'latitude', 'longitude']]
        for col in common_cols:
            df1[col] = df2[col].values

        ds3 = xr.open_dataset(folder + f + '/timeseries/' + f + '_era5_corrected_v1.nc')
        df3 = ds3.to_dataframe().reset_index()

        # Keep only rows in df2 with the same time as df1
        df3 = df3[df3['time'].isin(df1['time'])]
        df3 = df3[(df3['time'].dt.hour == 10)]
        df3_monthly = df3.resample('M', on='time').mean(numeric_only=True)
        df3_monthly = df3_monthly.iloc[:, 2:-2]
        df3_monthly = df3_monthly.add_suffix('_era')


        # Convert time column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df1['time']):
            df1['time'] = pd.to_datetime(df1['time'])

        # Keep only data at 1000
        df1 = df1[(df1['time'].dt.hour == 10)]
        # df1 = df1[df1['time'].dt.year >= 2013]

        # Resample to monthly mean
        df1_monthly = df1.resample('M', on='time').mean(numeric_only=True)
        df1_monthly['latitude'] = df2['latitude'].iloc[0]
        df1_monthly['longitude'] = df2['longitude'].iloc[0]
        df1_monthly['site'] = f  # Add site info
        df1_monthly = pd.concat([df1_monthly, df3_monthly], axis=1)

        # Collect for merging
        all_monthly.append(df1_monthly)

    # Merge all monthly dataframes into one final dataframe
    final_df = pd.concat(all_monthly)
    final_df.to_csv(current_dir+'/2_Output/urban_flux_data.csv')

    df3 = final_df[['SWdown', 'LWdown', 'Tair', 'SWup', 'LWup', 'Qle', 'Qh']]

    # Remove rows with any NaN values in df3
    df3 = df3.dropna()

    # Plot correlation heatmap for df3, handling NaN values
    plt.figure(figsize=(8, 6))
    corr = df3.corr(method='pearson', min_periods=1)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(f'Correlation Heatmap for {f}')
    plt.tight_layout()
    plt.show()
    plt.savefig('corr')








