import ee
import os
import geopandas as gpd

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-zhengfei')

from geeCodes import *
projection_modis = ee.ImageCollection("MODIS/061/MOD13A3").filterDate('2015-01-01','2016-01-01').first().projection()
# Define the FVC dataset
start_year = 2000
end_year = 2021
years = list(range(start_year, end_year + 1))

# 定义一个函数：对指定年份的数据做 Mosaic 合成，并重命名波段
def get_annual_mosaic(year):
    # 过滤年度数据
    annual_collection = ee.ImageCollection('WorldPop/GP/100m/pop') \
        .filter(ee.Filter.calendarRange(year, year, 'year')) \
        .select('population')

    # 执行 Mosaic 合成
    mosaic_image = annual_collection.mosaic() \
        .set('year', year) \
        .rename(f'pop_{year}')  # 以年份命名波段

    return mosaic_image

# 构建 ImageCollection：每年一个 Mosaic 图像
mosaic_images = ee.ImageCollection([get_annual_mosaic(y) for y in years])

# 合并为多波段图像
FVC0 = mosaic_images.toBands().setDefaultProjection(projection_modis)

resampledImage = FVC0.reduceResolution(
    reducer=ee.Reducer.mean(),
    maxPixels=50000
).reproject(
    crs='EPSG:4326',
    scale=1000
)
FVC = resampledImage 
# Load shapefile
current_dir = os.path.dirname(os.getcwd()).replace('\\', '/')
shapefile_path = current_dir + '/1_Input/shps/urban_cores_newtowns/urban_100km2.shp'
dfc = gpd.read_file(shapefile_path)
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']

# Export settings
folder = 'pop_500m'
for city in cities[1:]:  # Process the first city as an example
    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)
    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry)
    dscr = ('ntl_') + str(geo['ID'][0])
    print(f"Starting export for: {dscr}")
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

# Monitor tasks
# tasks = ee.batch.Task.list()
# for task in tasks:
#     print(task.status())