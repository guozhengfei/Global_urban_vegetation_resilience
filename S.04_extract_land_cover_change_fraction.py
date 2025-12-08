from geeCodes import *

import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize()

annual = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual")
projection_modis = ee.ImageCollection("MODIS/061/MOD09Q1").filterDate('2015-01-01','2016-01-01').first().projection().atScale(30)
annual_img = annual.mosaic()

# Define the old and new values for remapping
oldvalues = [10,11,12,20,51,52,61,62,71,72,81,82,91,92,120,121,122,130,140,181,182,183,184,185,186,187,190,150,152,153,200,201,202,210,220, 0]
newvalues = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 190, 7, 7, 7, 7, 7, 7, 8, 9, 0]
bnames = annual_img.bandNames()

# Define a custom function for remapping
def custom_fun(band):
    return annual_img.remap(oldvalues,newvalues,defaultValue= 0,bandName= band
    ).rename('b1')

# Modify the image collection using the custom function
modify_img = ee.ImageCollection(bnames.map(custom_fun))
img_p95 = modify_img.reduce(ee.Reducer.percentile([80]))
img_p05 = modify_img.reduce(ee.Reducer.percentile([20]))

# If img_p95 not equal img_p05, land cover changes
lcc_frac = img_p95.neq(img_p05).float().unmask(0) \
    .setDefaultProjection(projection_modis) \
    .reduceResolution(ee.Reducer.mean(), False, 600) \
    .reproject('EPSG:4326', None, 250)

FVC = lcc_frac

dfc = gpd.read_file('D:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/urban_100km2.shp')
# dfc = gpd.read_file('C:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/GUB_urban_cores.shp')
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']

folder = 'land_cover_change_250m'
for city in cities:#1087  1126
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    ####### GEE Geometry ###############################################
    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);
    dscr = ('lcc_')+str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=fvc,  # an ee.Image object.
                                     region=geometry,  # an ee.Geometry object.
                                     description=dscr,
                                     folder=folder,
                                     fileNamePrefix=dscr,
                                     scale=250,
                                     crs='epsg:4326')
    task.start()

# linearFit = FVC.select(['system:time_start', 'veg_frac']).reduce(ee.Reducer.linearFit())