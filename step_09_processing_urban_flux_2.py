import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import ee
ee.Authenticate()
ee.Initialize(project='ee-zhengfei')

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\', '/')
    df = pd.read_csv(current_dir+'/2_Output/urban_flux_data.csv')
    sites = list(set(df['site']))

    final_df = []
    for s in sites:
        df_s = df[df['site'] == s].copy()
        # Convert 'time' to datetime if not already
        df_s['time'] = pd.to_datetime(df_s['time'])
        # Change time from end of month to start of month
        df_s['time'] = df_s['time'].dt.to_period('M').dt.to_timestamp()
        lat = df_s['latitude'].iloc[0]
        lon = df_s['longitude'].iloc[0]
        time_strat = df_s['time'].values[0]
        time_strat = pd.to_datetime(time_strat).strftime('%Y-%m-%d')
        time_end = df_s['time'].values[-1]
        time_end = (pd.to_datetime(time_end) + pd.DateOffset(months=1)).strftime('%Y-%m-%d')

        ## LST
        imageC = ee.ImageCollection('MODIS/061/MOD11A2').select('LST_Day_1km').filterDate(time_strat, time_end).toBands()
        features = [ee.Feature(ee.Geometry.Point(lon, lat))]
        fc = ee.FeatureCollection(features)
        band_values = imageC.reduceRegions(fc, reducer=ee.Reducer.mean(),scale=1000).getInfo()

        values = list(band_values.values())
        val = values[2][0]
        records = []

        date = list(val.get('properties').keys())
        date = [x[:10] for x in date]
        data = list(val.get('properties').values())

        df0 = pd.DataFrame()
        # Specify the date format to avoid parsing errors
        df0['date'] = pd.to_datetime(date, format='%Y_%m_%d', errors='coerce')
        df0['value'] = np.array(data).astype(float)*0.02-273.15

        # Remove NaN values (including those from failed date parsing)
        df0 = df0.dropna(subset=['value', 'date'])

        # Group by year and month, calculate mean, and get the first day of each month
        monthly_lst = df0.groupby(df0['date'].dt.to_period('M')).agg({'value': 'mean'})
        monthly_lst['date'] = monthly_lst.index.to_timestamp()
        monthly_lst = monthly_lst.reset_index(drop=True)

        ## LW down
        imageC = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').select('surface_net_thermal_radiation_sum').filterDate(time_strat, time_end).toBands()
        features = [ee.Feature(ee.Geometry.Point(lon, lat))]
        fc = ee.FeatureCollection(features)
        band_values = imageC.reduceRegions(fc, reducer=ee.Reducer.mean(), scale=1000).getInfo()

        values = list(band_values.values())
        val = values[2][0]
        records = []

        date = list(val.get('properties').keys())
        date = [x[:6] for x in date]
        data = list(val.get('properties').values())

        df0 = pd.DataFrame()
        df0['date'] = pd.to_datetime(date, format='%Y%m', errors='coerce')
        df0['value'] = np.array(data).astype(float)/(3600*24*30)
        df0 = df0.dropna(subset=['value', 'date'])

        # Group by year and month, calculate mean, and get the first day of each month
        monthly_lw = df0.groupby(df0['date'].dt.to_period('M')).agg({'value': 'mean'})
        monthly_lw['date'] = monthly_lw.index.to_timestamp()
        monthly_lw = monthly_lw.reset_index(drop=True)

        ## Rn
        imageC = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').select(
            'surface_net_solar_radiation_sum').filterDate(time_strat, time_end).toBands()
        features = [ee.Feature(ee.Geometry.Point(lon, lat))]
        fc = ee.FeatureCollection(features)
        band_values = imageC.reduceRegions(fc, reducer=ee.Reducer.mean(), scale=1000).getInfo()

        values = list(band_values.values())
        val = values[2][0]
        records = []

        date = list(val.get('properties').keys())
        date = [x[:6] for x in date]
        data = list(val.get('properties').values())

        df0 = pd.DataFrame()
        df0['date'] = pd.to_datetime(date, format='%Y%m', errors='coerce')
        df0['value'] = np.array(data).astype(float) / (3600 * 24 * 30)
        df0 = df0.dropna(subset=['value', 'date'])

        monthly_rn = df0.groupby(df0['date'].dt.to_period('M')).agg({'value': 'mean'})
        monthly_rn['date'] = monthly_rn.index.to_timestamp()
        monthly_rn = monthly_rn.reset_index(drop=True)

        ## NDVI
        imageC = ee.ImageCollection('MODIS/061/MOD13Q1').select('EVI').filterDate(time_strat,time_end).toBands()
        features = [ee.Feature(ee.Geometry.Point(lon, lat))]
        fc = ee.FeatureCollection(features)
        band_values = imageC.reduceRegions(fc, reducer=ee.Reducer.mean(), scale=250).getInfo()

        values = list(band_values.values())
        val = values[2][0]
        records = []

        date = list(val.get('properties').keys())
        date = [x[:10] for x in date]
        data = list(val.get('properties').values())

        df0 = pd.DataFrame()
        # Specify the date format to avoid parsing errors
        df0['date'] = pd.to_datetime(date, format='%Y_%m_%d', errors='coerce')
        df0['value'] = np.array(data).astype(float) * 0.0001

        # Remove NaN values (including those from failed date parsing)
        df0 = df0.dropna(subset=['value', 'date'])

        # Group by year and month, calculate mean, and get the first day of each month
        monthly_ndvi = df0.groupby(df0['date'].dt.to_period('M')).agg({'value': 'max'})
        monthly_ndvi['date'] = monthly_ndvi.index.to_timestamp()
        monthly_ndvi = monthly_ndvi.reset_index(drop=True)

        ## Tair
        imageC = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').select('temperature_2m').filterDate(time_strat, time_end).toBands()
        features = [ee.Feature(ee.Geometry.Point(lon, lat))]
        fc = ee.FeatureCollection(features)
        band_values = imageC.reduceRegions(fc, reducer=ee.Reducer.mean(), scale=1000).getInfo()

        values = list(band_values.values())
        val = values[2][0]
        records = []

        date = list(val.get('properties').keys())
        date = [x[:6] for x in date]
        data = list(val.get('properties').values())

        df0 = pd.DataFrame()
        # Specify the date format to avoid parsing errors
        df0['date'] = pd.to_datetime(date, format='%Y%m', errors='coerce')
        df0['value'] = np.array(data).astype(float)

        # Remove NaN values (including those from failed date parsing)
        df0 = df0.dropna(subset=['value', 'date'])

        # Group by year and month, calculate mean, and get the first day of each month
        monthly_ta = df0.groupby(df0['date'].dt.to_period('M')).agg({'value': 'max'})
        monthly_ta['date'] = monthly_ta.index.to_timestamp()
        monthly_ta = monthly_ta.reset_index(drop=True)

        ## vpd
        imageC = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').select('vpd').filterDate(time_strat, time_end).toBands()
        features = [ee.Feature(ee.Geometry.Point(lon, lat))]
        fc = ee.FeatureCollection(features)
        band_values = imageC.reduceRegions(fc, reducer=ee.Reducer.mean(), scale=1000).getInfo()

        values = list(band_values.values())
        val = values[2][0]
        records = []

        date = list(val.get('properties').keys())
        date = [x[:6] for x in date]
        data = list(val.get('properties').values())

        df0 = pd.DataFrame()
        # Specify the date format to avoid parsing errors
        df0['date'] = pd.to_datetime(date, format='%Y%m', errors='coerce')
        df0['value'] = np.array(data).astype(float) * 0.01

        # Remove NaN values (including those from failed date parsing)
        df0 = df0.dropna(subset=['value', 'date'])

        # Group by year and month, calculate mean, and get the first day of each month
        monthly_vpd = df0.groupby(df0['date'].dt.to_period('M')).agg({'value': 'max'})
        monthly_vpd['date'] = monthly_vpd.index.to_timestamp()
        monthly_vpd = monthly_vpd.reset_index(drop=True)



        df_merge_i = monthly_lst[['date', 'value']].rename(columns={'value': 'lst'}).merge(
            monthly_lw[['date', 'value']].rename(columns={'value': 'lw_net'}), on='date', how='outer').merge(
            monthly_rn[['date', 'value']].rename(columns={'value': 'sw_net'}), on='date', how='outer').merge(
            monthly_ndvi[['date', 'value']].rename(columns={'value': 'ndvi'}), on='date', how='outer').merge(
            monthly_vpd[['date', 'value']].rename(columns={'value': 'vpd'}), on='date', how='outer').merge(
            monthly_ta[['date', 'value']].rename(columns={'value': 'ta_rs'}), on='date', how='outer')

        df_s = df_s.rename(columns={'time': 'date'}).merge(df_merge_i, on='date')
        final_df.append(df_s)
        print(s)

    # 合并所有站点数据
    final_df = pd.concat(final_df, ignore_index=True)
    print(final_df.head())
    final_df.to_csv(current_dir+'/2_Output/flux_and_rs.csv')














