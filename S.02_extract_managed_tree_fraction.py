from geeCodes import *
import ee
import os
import geopandas as gpd

# Authenticate and Initialize Earth Engine
# ee.Authenticate() # Uncomment if you need to re-authenticate
ee.Initialize(project='ee-zhengfei')

# -------------------------------------------------------------------------
# 1. Define the 500m Managed Tree Fraction for 2020
# -------------------------------------------------------------------------
# Forest Typology (ForTy) 2020 v1.0 classes:
# 4 = Plantation Forest
# 5 = Tree Crops and Agroforestry
forty_coll = ee.ImageCollection(
    "projects/nature-trace/assets/forest_typology/forest_typology_2020_v1_0_collection"
)

# Important:
# Get native projection from one original image, not from the mosaicked image
source_projection = ee.Projection('EPSG:4326').atScale(30)

# Mosaic and immediately assign a valid default projection
forty_2020 = (
    forty_coll
    .mosaic()
    .select([0, 1, 2, 3, 4])
    .setDefaultProjection(source_projection)
)

# Compute class 6: Other land = 250 - sum(classes 1-5)
sum_classes = (
    forty_2020
    .select([0, 1, 2, 3, 4])
    .reduce(ee.Reducer.sum())
)

other_land = (
    ee.Image.constant(250)
    .subtract(sum_classes)
    .rename('other_land')
)

# Argmax among classes 1-6
classified = (
    forty_2020
    .addBands(other_land)
    .toArray()
    .arrayArgmax()
    .arrayGet([0])
    .add(1)
    .rename('forest_typology_class')
)

# Managed tree classes: 4 and 5
managed_tree_mask = (
    classified
    .remap([4, 5], [1, 1], 0)
    .rename('managed_tree')
    .float()
)

# Aggregate binary mask to 500 m fraction
managed_tree_frac = (
    managed_tree_mask
    .reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=65536
    )
    .reproject(
        crs='EPSG:4326',
        scale=500
    )
    .rename('managed_tree_fraction')
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
folder = 'managedTreeCover_500m'

# -------------------------------------------------------------------------
# 3. Iterate through cities and Export
# -------------------------------------------------------------------------
for city in cities:
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    ####### GEE Geometry ###############################################
    geometry = returnGeometry(S[0])

    # Clip the 500m fraction image to the city geometry
    fvc = managed_tree_frac.clip(geometry)

    # Update description prefix for managed tree cover
    dscr = f"managedTreeC_{str(geo['ID'][0])}"
    print(f"Starting export task for: {dscr}")

    task = ee.batch.Export.image.toDrive(
        image=fvc,
        region=geometry,
        description=dscr,
        folder=folder,
        fileNamePrefix=dscr,
        scale=500,
        crs='EPSG:4326',
        maxPixels=1e13  # Prevents errors on large geometry bounding boxes
    )
    task.start()
