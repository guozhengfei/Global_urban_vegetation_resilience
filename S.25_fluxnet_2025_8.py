# export urban flux site ndvi: 3km*3km
import ee
import pandas as pd
from datetime import datetime
import os
# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-zhengfei')

def export_ndvi_as_tiff(df, output_folder='landsat_ndvi'):
    """
    Export Landsat LST as TIFF images for multiple points with 3km×3km boundaries

    Args:
        df: DataFrame with columns 'lat', 'lon', and 'sitename'
        output_folder: Google Drive folder for output files

    Returns:
        List of export task information
    """
    # Validate input DataFrame
    required_columns = {'lat', 'lon', 'site'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    tasks = []

    for _, row in df.iterrows():
        # Create 3km×3km boundary (2500m radius buffer)
        point = ee.Geometry.Point(row['lon'], row['lat'])
        boundary = point.buffer(2500).bounds()

        # Landsat 8 Collection 2, Tier 1
        landsat = ee.ImageCollection('LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI') \
            .filterDate('2015-01-01', '2024-12-31') \
            .select('NDVI') \
            .filterBounds(boundary) \

        # Calculate monthly medians
        monthly_images = []
        for month in range(1, 13):
            monthly_col = landsat.filter(ee.Filter.calendarRange(month, month, 'month'))
            monthly_median = monthly_col.median().rename(f'NDVI_{month:02d}')
            monthly_images.append(monthly_median)

        # Combine all monthly images
        final_image = ee.Image.cat(monthly_images).clip(boundary)

        # Export parameters
        task = ee.batch.Export.image.toDrive(
            image=final_image,
            description=f"NDVI_{row['site']}",
            folder=output_folder,
            fileNamePrefix=f"NDVI_{row['site']}",
            region=boundary,
            scale=30,  # Landsat native resolution
            crs='EPSG:4326',
            fileFormat='GeoTIFF',
            maxPixels=1e13
        )
        task.start()
        tasks.append({
            'site_id': row['site'],
            'task_id': task.id,
            'status': task.status()
        })

    return tasks

# Example usage with test data
# Example usage
if __name__ == '__main__':
    final_df = pd.read_csv(os.path.join('..', '2_Output', 'flux_and_rs.csv'))
    points_df = final_df[['lat','lon','site']]
    points_df = points_df.groupby('site',as_index=False).aggregate('mean')

    tasks = export_ndvi_as_tiff(points_df)
    print("Export tasks started:")
    for task in tasks:
        print(f"Site: {task['site_id']} | Task ID: {task['task_id']} | Status: {task['status']['state']}")
