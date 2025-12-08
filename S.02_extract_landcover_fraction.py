from geeCodes import *

import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize()

FVC = (ee.ImageCollection('ESA/WorldCover/v200').first()
    .select('Map')
    .eq(80)
    .reduce('mean')
    .float()
    .unmask(0)
    .reduceResolution(ee.Reducer.mean(), False, 12000)
    .reproject('EPSG:4326', None, 250)
) # 10:tree; 40:crop; 30: grass; 50 biult-up; 80 water

current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

dfc = gpd.read_file(current_dir + '/1_Input/shps/urban_cores_newtowns/urban_100km2.shp')
# dfc = gpd.read_file('C:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/GUB_urban_cores.shp')
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']

folder = 'waterCover_250m'
for city in cities:#1087  1126
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    ####### GEE Geometry ###############################################
    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);
    dscr = ('waterC_')+str(geo['ID'][0])
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