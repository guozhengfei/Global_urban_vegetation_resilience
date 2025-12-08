import ee
import os
import geopandas as gpd

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-zhengfei')

from geeCodes import *

# Define the FVC dataset
FVC = (ee.ImageCollection('NASA/VIIRS/002/VNP46A2')
       .filter(ee.Filter.date('2013-01-01', '2015-01-01'))
       .select('Gap_Filled_DNB_BRDF_Corrected_NTL')
       .mean())

# Load shapefile
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
shapefile_path = current_dir + '/1_Input/shps/urban_cores_newtowns/urban_100km2.shp'
dfc = gpd.read_file(shapefile_path)
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']

# Export settings
folder = 'NTL_500m'
for city in cities[1:]:  # Process the first city as an example
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)
    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry)
    dscr = ('ntl_') + str(geo['ID'][0])
    print(f"Starting export for: {dscr}")
    task = ee.batch.Export.image.toDrive(
        image=fvc,
        region=geometry,
        description=dscr,
        folder=folder,
        fileNamePrefix=dscr,
        scale=1000,
        crs='epsg:4326'
    )
    task.start()

# Monitor tasks
# tasks = ee.batch.Task.list()
# for task in tasks:
#     print(task.status())