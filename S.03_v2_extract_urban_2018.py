from geeCodes import *
import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize()

water_mask = ee.Image(1).subtract(ee.Image('MERIT/Hydro/v1_0_1').select('wat'))
FVC = ee.Image("projects/ee-zhengfei/assets/urban_both_raster")
dfc = gpd.read_file('D:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/urban_100km2.shp')
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']

folder = 'urban_2018_250m'
for city in cities:#1087
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)
    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);

    dscr = 'urban_'+str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=fvc,  # an ee.Image object.
                                     region=geometry,  # an ee.Geometry object.
                                     description=dscr,
                                     folder=folder,
                                     fileNamePrefix=dscr,
                                     scale = 250,
                                     crs = 'epsg:4326')
    task.start()