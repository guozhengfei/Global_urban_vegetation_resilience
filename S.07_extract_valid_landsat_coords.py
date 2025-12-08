from geeCodes import *

import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize()

# Load the image collections and images
world_cover = ee.ImageCollection('ESA/WorldCover/v200').first()
gisd30_image = ee.Image("projects/sat-io/open-datasets/GISD30_1985_2020")

# Calculate vegetation natural cover
veg_nat_cover = world_cover.lte(30) \
    .reduce('mean') \
    .float() \
    .unmask(0) \
    .reduceResolution(ee.Reducer.mean(), False, 12000) \
    .reproject('EPSG:4326', None, 50)

# Calculate crop cover
crop_cover = world_cover.eq(40) \
    .reduce('mean') \
    .float() \
    .unmask(0) \
    .reduceResolution(ee.Reducer.mean(), False, 12000) \
    .reproject('EPSG:4326', None, 50)

# Calculate GISD30
gisd30 = gisd30_image.select('b1') \
    .gt(4) \
    .reduce('mean') \
    .float() \
    .unmask(0) \
    .reduceResolution(ee.Reducer.mean(), False, 12000) \
    .reproject('EPSG:4326', None, 50)

# Calculate the difference between veg_nat_cover and crop_cover
diff_cover = veg_nat_cover.add(crop_cover.multiply(-1))

# Create the valid image based on the conditions
valid_image = diff_cover.gt(0.01).And(crop_cover.lt(0.2)).And(gisd30.lt(0.2))

FVC = valid_image

current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

dfc = gpd.read_file(current_dir + '/1_Input/shps/urban_cores_newtowns/urban_100km2.shp')
# dfc = gpd.read_file('C:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/GUB_urban_cores.shp')
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']

folder = 'valid_area_landsat'
for city in cities[1:]:#1087  1126
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    ####### GEE Geometry ###############################################
    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);
    dscr = ('valid_area_')+str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=fvc,  # an ee.Image object.
                                     region=geometry,  # an ee.Geometry object.
                                     description=dscr,
                                     folder=folder,
                                     fileNamePrefix=dscr,
                                     scale=50,
                                     crs='epsg:4326')
    task.start()

# linearFit = FVC.select(['system:time_start', 'veg_frac']).reduce(ee.Reducer.linearFit())