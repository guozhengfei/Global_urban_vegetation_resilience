from geeCodes import *
import ee
ee.Authenticate()
ee.Initialize()

shp = ee.FeatureCollection("projects/ee-zhengfei/assets/urban_2000")
image = ee.Image.constant(1)
raster = image.clipToCollection(shp)

dfc = gpd.read_file('D:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/urban_100km2.shp')
# dfc = gpd.read_file('C:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/GUB_urban_cores.shp')
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']

folder = 'urban_2000_250m'
for city in cities[1:]:#1087  1126
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    ####### GEE Geometry ###############################################
    geometry = returnGeometry(S[0])
    fvc = raster.clip(geometry);
    dscr = ('urban_2000_')+str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=fvc,  # an ee.Image object.
                                     region=geometry,  # an ee.Geometry object.
                                     description=dscr,
                                     folder=folder,
                                     fileNamePrefix=dscr,
                                     scale=250,
                                     crs='epsg:4326')
    task.start()