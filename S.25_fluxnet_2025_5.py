import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import ee

# Initialize Earth Engine
def initialize_earth_engine():
    """Initialize Google Earth Engine with authentication."""
    ee.Authenticate()
    ee.Initialize(project='ee-zhengfei')


def load_site_data():
    """Load and preprocess site data from CSV."""
    csv_path = os.path.join('..', '..', 'Fluxnet2025', 'urban_flux_30_sites.csv')
    df = pd.read_csv(csv_path.replace('\\', '/'))
    df['time'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01',
                                format='%Y-%m-%d')
    return df, list(set(df['site']))


def get_date_range(df_site):
    """Get start and end dates for a site."""
    time_start = pd.to_datetime(df_site['time'].values[0]).strftime('%Y-%m-%d')
    time_end = (pd.to_datetime(df_site['time'].values[-1]) +
                pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    return time_start, time_end


def fetch_earth_engine_data(collection, bands, point, date_range, scale,
                            value_processor=None, date_format='%Y_%m_%d'):
    """
    Fetch data from Earth Engine collections.

    Args:
        collection: Earth Engine collection ID
        bands: List of bands to select
        point: (lon, lat) coordinates
        date_range: (start_date, end_date)
        scale: Resolution in meters
        value_processor: Function to process raw values
        date_format: Format of dates in properties

    Returns:
        DataFrame with processed data
    """
    image_collection = (ee.ImageCollection(collection)
                        .select(bands)
                        .filterDate(*date_range)
                        .toBands())

    features = [ee.Feature(ee.Geometry.Point(*point))]
    fc = ee.FeatureCollection(features)

    band_values = image_collection.reduceRegions(
        fc, reducer=ee.Reducer.mean(), scale=scale).getInfo()

    values = list(band_values.values())
    val = values[2][0]

    dates = [''.join(k.split('_'))[:6] for k in val.get('properties').keys()]
    data = list(val.get('properties').values())

    df = pd.DataFrame({
        'date': pd.to_datetime(dates, format='%Y%m', errors='coerce'),
        'value': np.array(data).astype(float)
    })

    if value_processor:
        df['value'] = value_processor(df['value'])

    # df = df.dropna(subset=['value', 'date'])

    # Group by month and calculate mean
    monthly = df.groupby(df['date'].dt.to_period('M')).agg({'value': 'mean'})
    monthly['date'] = monthly.index.to_timestamp()
    return monthly.reset_index(drop=True)


def process_site_data(df_site):
    """Process data for a single site."""
    lat, lon = df_site['lat'].iloc[0], df_site['lon'].iloc[0]
    time_start, time_end = get_date_range(df_site)

    # Process LST data
    monthly_lstD = fetch_earth_engine_data(
        'MODIS/061/MOD11A2', ['LST_Day_1km'],
        (lon, lat), (time_start, time_end), 1000,
        lambda x: x * 0.02 - 273.15
    )

    monthly_lstN = fetch_earth_engine_data(
        'MODIS/061/MOD11A2', ['LST_Night_1km'],
        (lon, lat), (time_start, time_end), 1000,
        lambda x: x * 0.02 - 273.15
    )

    # Process SWdown data
    monthly_sw_net = fetch_earth_engine_data(
        'ECMWF/ERA5_LAND/MONTHLY_AGGR', ['surface_solar_radiation_downwards_sum'],
        (lon, lat), (time_start, time_end), 1000,
        lambda x: x / (3600 * 24 * 30),
        date_format='%Y%m'
    )

    # Process LWnet data
    monthly_lw = fetch_earth_engine_data(
        'ECMWF/ERA5_LAND/MONTHLY_AGGR', ['surface_net_thermal_radiation_sum'],
        (lon, lat), (time_start, time_end), 1000,
        lambda x: x / (3600 * 24 * 30),
        date_format='%Y%m'
    )

    # Process NDVI data
    monthly_ndvi = fetch_earth_engine_data(
        'MODIS/061/MOD13Q1', ['EVI'],
        (lon, lat), (time_start, time_end), 250,
        lambda x: x * 0.0001
    )

    # Process Air Temperature data mean
    monthly_ta = fetch_earth_engine_data(
        'ECMWF/ERA5_LAND/MONTHLY_AGGR', ['temperature_2m'],
        (lon, lat), (time_start, time_end), 1000,
        date_format='%Y%m'
    )

    # Process Air Temperature data max
    monthly_ta_max = fetch_earth_engine_data(
        'ECMWF/ERA5_LAND/MONTHLY_AGGR', ['temperature_2m_max'],
        (lon, lat), (time_start, time_end), 1000,
        date_format='%Y%m'
    )

    # Process Air Temperature data min
    monthly_ta_min = fetch_earth_engine_data(
        'ECMWF/ERA5_LAND/MONTHLY_AGGR', ['temperature_2m_min'],
        (lon, lat), (time_start, time_end), 1000,
        date_format='%Y%m'
    )

    # Process dewpoint Temperature data
    monthly_td = fetch_earth_engine_data(
        'ECMWF/ERA5_LAND/MONTHLY_AGGR', ['dewpoint_temperature_2m'],
        (lon, lat), (time_start, time_end), 1000,
        date_format='%Y%m'
    )

    # Merge all data
    merged_data = (
        monthly_lstD[['date', 'value']].rename(columns={'value': 'lstD'})
        .merge(monthly_lstN[['date', 'value']].rename(columns={'value': 'lstN'}),
               on='date', how='outer')
        .merge(monthly_sw_net[['date', 'value']].rename(columns={'value': 'sw_down'}),
               on='date', how='outer')
        .merge(monthly_lw[['date', 'value']].rename(columns={'value': 'lw_net'}),
               on='date', how='outer')
        .merge(monthly_ndvi[['date', 'value']].rename(columns={'value': 'ndvi'}),
               on='date', how='outer')
        .merge(monthly_ta[['date', 'value']].rename(columns={'value': 'ta_rs'}),
               on='date', how='outer')
        .merge(monthly_ta_max[['date', 'value']].rename(columns={'value': 'ta_max_rs'}),
               on='date', how='outer')
        .merge(monthly_ta_min[['date', 'value']].rename(columns={'value': 'ta_rs_min'}),
               on='date', how='outer')
        .merge(monthly_td[['date', 'value']].rename(columns={'value': 'td_rs'}),
               on='date', how='outer')
    )

    return df_site.rename(columns={'time': 'date'}).merge(merged_data, on='date')


def main():
    """Main processing pipeline."""
    initialize_earth_engine()
    df, sites = load_site_data()

    final_data = []
    for site in sites:
        site_df = df[df['site'] == site].copy()
        processed_data = process_site_data(site_df)
        final_data.append(processed_data)
        print(f"Processed site: {site}")

    # Combine all site data
    final_df = pd.concat(final_data, ignore_index=True)
    print(final_df.head())

    # Save results
    output_path = os.path.join('..', '2_Output', 'flux_and_rs.csv')
    final_df.to_csv(output_path.replace('\\', '/'))


if __name__ == '__main__':
    main()
