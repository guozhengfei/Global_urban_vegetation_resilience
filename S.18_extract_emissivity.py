from geeCodes import *

import os
import time
import ee
import geopandas as gpd

# ---------------------------------------------------------------------
# Earth Engine initialization
# ---------------------------------------------------------------------
try:
    ee.Initialize(project='ee-zhengfei')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='ee-zhengfei')


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
START_DATE = '2001-01-01'
END_DATE = '2024-01-01'   # exclusive, includes all months from 2001 to 2023

MOD21_COLLECTION = 'MODIS/061/MOD21C3'

DRIVE_FOLDER = 'MOD21C3_emissivity_2001_2023_mean'

# MOD21C3 nominal pixel size in Earth Engine catalog
EXPORT_SCALE = 1000
EXPORT_CRS = 'EPSG:4326'

EMIS_BANDS = [
    'Emis_29_Day',
    'Emis_31_Day',
    'Emis_32_Day',
    'Emis_29_Night',
    'Emis_31_Night',
    'Emis_32_Night'
]


# ---------------------------------------------------------------------
# Build one long-term mean emissivity image for 2001–2023
# ---------------------------------------------------------------------
mod21 = (
    ee.ImageCollection(MOD21_COLLECTION)
    .filterDate(START_DATE, END_DATE)
    .select(EMIS_BANDS)
)

# First average across emissivity bands within each monthly image,
# then average all monthly images from 2001–2023.
emis_mean_2001_2023 = (
    mod21
    .map(lambda img: img.reduce(ee.Reducer.mean()).rename('emis_mean'))
    .mean()
    .rename('emis_mean_2001_2023')
)

# Optional: keep physically plausible emissivity values only.
# You can remove this line if you want absolutely no filtering.
emis_mean_2001_2023 = emis_mean_2001_2023.updateMask(
    emis_mean_2001_2023.gte(0.49).And(emis_mean_2001_2023.lte(1.00))
)


# ---------------------------------------------------------------------
# Read city shapefile
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

shp_path = current_dir + '/1_Input/shps/urban_cores_newtowns/urban_100km2.shp'

dfc = gpd.read_file(shp_path)
dfc = dfc.to_crs('EPSG:4326')

cities = dfc['ID'].unique()


# ---------------------------------------------------------------------
# Export one single-band GeoTIFF per city
# ---------------------------------------------------------------------
for city in cities:
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    city_id = str(geo['ID'].iloc[0])

    S = returnCityBoundary(geo)

    # GEE geometry
    geometry = returnGeometry(S[0])

    img_export = emis_mean_2001_2023.clip(geometry)

    dscr = f'MOD21C3_emis_mean_2001_2023_{city_id}'
    print(dscr)

    task = ee.batch.Export.image.toDrive(
        image=img_export,
        region=geometry,
        description=dscr,
        folder=DRIVE_FOLDER,
        fileNamePrefix=dscr,
        scale=EXPORT_SCALE,
        crs=EXPORT_CRS,
        maxPixels=1e13,
        fileFormat='GeoTIFF',
        skipEmptyTiles=True
    )

    task.start()

    # Avoid submitting too many tasks too quickly
    time.sleep(0.2)