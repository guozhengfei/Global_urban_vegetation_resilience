from geeCodes import *

import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize(project='ee-zhengfei')

imagecollections = []
for n in range(0, 23*12):
    start = ee.Date('2001-01-01').advance(n, 'month')
    end = start.advance(1, 'month')
    evi = ((ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
              .filter(ee.Filter.date(start, end)))
              .select('soil')).mosaic();
    imagecollections.append(evi)

FVC = ee.ImageCollection.fromImages(imagecollections).select('soil').toBands()

dfc = gpd.read_file('/Volumes/Zhengfei_01/project 3 urban resilience/1_Input/shps/urban_cores_newtowns/urban_100km2.shp')

dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']
folder = 'SM_Terraclimate_urban'

for city in cities:
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);

    dscr = 'SM_monthly_'+str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=fvc,  # an ee.Image object.
                                     region=geometry,  # an ee.Geometry object.
                                     description=dscr,
                                     folder=folder,
                                     fileNamePrefix=dscr,
                                     scale = 8000,
                                     crs = 'epsg:4326')
    task.start()
