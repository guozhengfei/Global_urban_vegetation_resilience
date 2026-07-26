from geeCodes import *

import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize(project='ee-zhengfei')

imagecollections = []
for year in [2006, 2016, 2023]:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, 'year')
    landcover = (ee.ImageCollection('MODIS/061/MCD12Q1')
                 .filter(ee.Filter.date(start, end))
                 .select('LC_Type4')
                 .first()
                 .rename('LC_Type4_' + str(year)))
    imagecollections.append(landcover)

FVC = ee.Image.cat(imagecollections)

dfc = gpd.read_file('/Volumes/Zhengfei_01/project 3 urban resilience/1_Input/shps/urban_cores_newtowns/urban_100km2.shp')

dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']
folder = 'landcover_500m'

for city in cities[1:]:
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);

    dscr = 'landcover_500m_'+str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=fvc,  # an ee.Image object.
                                     region=geometry,  # an ee.Geometry object.
                                     description=dscr,
                                     folder=folder,
                                     fileNamePrefix=dscr,
                                     scale = 500,
                                     crs = 'epsg:4326')
    task.start()

# tasks = ee.batch.Task.list()
# tasks[0].status()

# for task in ee.batch.Task.list():
# 	task.cancel()

# import tifffile as tf
# data = tf.imread('/Users/zhengfei/Downloads/ndvi_monthly_1.0.tif')
# plt.figure(); plt.imshow(data[:,:,1])