from geeCodes import *

import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize()


imagecollections = []
for n in range(0, 23*12):
    start = ee.Date('2001-01-01').advance(n, 'month')
    end = start.advance(1, 'month')
    evi = ((ee.ImageCollection('MODIS/061/MOD13A3')
              .filter(ee.Filter.date(start, end)))
              .select('EVI')).mosaic();
    imagecollections.append(evi)

FVC = ee.ImageCollection.fromImages(imagecollections).select('EVI').toBands()

dfc = gpd.read_file('D:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/urban_100km2.shp')

dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']
folder = 'evi_monthly_1km'

for city in cities[2:]:
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);

    dscr = 'evi_monthly_'+str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=fvc,  # an ee.Image object.
                                     region=geometry,  # an ee.Geometry object.
                                     description=dscr,
                                     folder=folder,
                                     fileNamePrefix=dscr,
                                     scale = 1000,
                                     crs = 'epsg:4326')
    task.start()