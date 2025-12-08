from geeCodes import *

import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize(project='ee-zhengfei')

# uhi = ee.ImageCollection('YALE/YCEO/UHI/UHI_yearly_pixel/v4').select('Daytime').toBands()
uhi = ee.ImageCollection('YALE/YCEO/UHI/UHI_yearly_pixel/v4').select('Nighttime').toBands()

# Resample the masked image using reduceResolution and reproject
resampledImage = uhi.reduceResolution(
    reducer=ee.Reducer.mean(),
    maxPixels=1000
).reproject(
    crs='EPSG:4326',
    scale=1000
)
FVC = resampledImage # 10:tree; 40:crop; 30: grass; 50 biult-up; 80 water

current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')

dfc = gpd.read_file(current_dir + '/1_Input/shps/urban_cores_newtowns/urban_100km2.shp')
# dfc = gpd.read_file('C:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/GUB_urban_cores.shp')
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']
folder = 'uhi_1000m_night'

for city in cities:#1087  1126
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    ####### GEE Geometry ###############################################
    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);
    dscr = ('uhi_')+str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(
        image=fvc,
        region=geometry,
        description=dscr,
        folder=folder,
        fileNamePrefix=dscr,
        scale=1000,
        crs='epsg:4326'
    )
    task.start()

# linearFit = FVC.select(['system:time_start', 'veg_frac']).reduce(ee.Reducer.linearFit())