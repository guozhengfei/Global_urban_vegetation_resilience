from geeCodes import *
import ee
import os
import geopandas as gpd

# Authenticate and Initialize Earth Engine
# ee.Authenticate() # Uncomment if you need to re-authenticate
ee.Initialize(project='ee-zhengfei')

# -------------------------------------------------------------------------
# 1. Define the 500m Cultivated Grassland Fraction for 2020
# -------------------------------------------------------------------------
grassland_coll = ee.ImageCollection("projects/global-pasture-watch/assets/ggc-30m/v1/grassland_c")

# Filter for the year 2020 and get the first image
grassland_2020 = grassland_coll.filterDate('2020-01-01', '2021-01-01').first()

# Calculate the fraction at 500m resolution
# By unmasking to 0 and reducing via mean, the 30m binary/probabilistic
# pixels are averaged into a 500m fractional cover map.

grassland_frac = (grassland_2020
    .select('dominant_class')  # Select the classification band
    .eq(1)                     # 1 = Cultivated grassland, 0 = Other/Natural
    .float()                   # Convert to float for accurate mean reduction
    .unmask(0)                 # Ensure unmapped areas are treated as 0 fraction
    .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=65536)
    .reproject(crs='EPSG:4326', scale=500)
)
# -------------------------------------------------------------------------
# 2. Setup Local Directories and Read Shapefile
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
shp_path = f"{current_dir}/1_Input/shps/urban_cores_newtowns/urban_100km2.shp"

dfc = gpd.read_file(shp_path)
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']

# Updated folder name for Google Drive
folder = 'Managed_grasslandFraction_500m'

# -------------------------------------------------------------------------
# 3. Iterate through cities and Export
# -------------------------------------------------------------------------
for city in cities:
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    ####### GEE Geometry ###############################################
    geometry = returnGeometry(S[0])

    # Clip the 500m fraction image to the city geometry
    fvc = grassland_frac.clip(geometry)

    # Update description prefix for grassland
    dscr = f"managed_grassland_fraction_{str(geo['ID'][0])}"
    print(f"Starting export task for: {dscr}")

    task = ee.batch.Export.image.toDrive(
        image=fvc,
        region=geometry,
        description=dscr,
        folder=folder,
        fileNamePrefix=dscr,
        scale=500,
        crs='EPSG:4326',
        maxPixels=1e13  # Added to prevent errors on larger geometries
    )
    task.start()