from geeCodes import *

import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize(project='ee-zhengfei')

canopy_height = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1")
dataset = ee.ImageCollection('ESA/WorldCover/v200').first()
mask = dataset.gte(0).And(dataset.lte(10)) # remove all the nontree pixel
maskedImage = canopy_height.updateMask(mask)

# Resample the masked image using reduceResolution and reproject
resampledImage = maskedImage.reduceResolution(
    reducer=ee.Reducer.mean(),
    maxPixels=1000
).reproject(
    crs='EPSG:4326',
    scale=250
)
FVC = resampledImage # 10:tree; 40:crop; 30: grass; 50 biult-up; 80 water

current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

dfc = gpd.read_file(current_dir + '/1_Input/shps/urban_cores_newtowns/urban_100km2.shp')
# dfc = gpd.read_file('C:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/GUB_urban_cores.shp')
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']

folder = 'canopy_height_250m'


# Get folder ID and use it in exports
folder_id = '1zu5H4mnY4KOD32WG6uXy5r5Ja1cMgZzh?usp=drive_link'

for city in cities:#1087  1126
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    ####### GEE Geometry ###############################################
    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);
    dscr = 'canopyHeight_'+str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=fvc,  # an ee.Image object.
                                     region=geometry,  # an ee.Geometry object.
                                     description=dscr,
                                     folder=folder_id,
                                     fileNamePrefix=dscr,
                                     scale=250,
                                     crs='epsg:4326')
    task.start()

# linearFit = FVC.select(['system:time_start', 'veg_frac']).reduce(ee.Reducer.linearFit())