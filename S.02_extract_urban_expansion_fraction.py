from geeCodes import *

import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize()

FVC = ee.Image("projects/sat-io/open-datasets/GISD30_1985_2020").select('b1').gt(4).reduce('mean').float().unmask(0).reduceResolution(ee.Reducer.mean(), False, 12000).reproject('EPSG:4326', None, 500)

current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

dfc = gpd.read_file(current_dir + '/1_Input/shps/urban_cores_newtowns/urban_100km2.shp')
# dfc = gpd.read_file('C:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/GUB_urban_cores.shp')
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']

folder = 'urban_expansion_frac_500m'
for city in cities:#1087  1126
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    ####### GEE Geometry ###############################################
    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);
    dscr = ('urban_exp_C_')+str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=fvc,  # an ee.Image object.
                                     region=geometry,  # an ee.Geometry object.
                                     description=dscr,
                                     folder=folder,
                                     fileNamePrefix=dscr,
                                     scale=500,
                                     crs='epsg:4326')
    task.start()

# linearFit = FVC.select(['system:time_start', 'veg_frac']).reduce(ee.Reducer.linearFit())