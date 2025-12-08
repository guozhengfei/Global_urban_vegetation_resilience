from geeCodes import *

import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize()

imagecollections = []
for n in range(0, 23*12*2):
    start = ee.Date('2001-01-01').advance(n*0.5, 'month')
    end = start.advance(0.5, 'month')
    evi = ((ee.ImageCollection('MODIS/MCD43A4_006_NDVI')
              .filter(ee.Filter.date(start, end)))
              .select('NDVI')).median();
    imagecollections.append(evi)

FVC = ee.ImageCollection.fromImages(imagecollections).toBands()#.select('NDVI')

dfc = gpd.read_file('D:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/urban_100km2.shp')

dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']
folder = 'nadir_ndvi_15d_500'

for city in cities[1:]:
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);

    dscr = 'nadir_ndvi_15d_'+str(geo['ID'][0])
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