from modis_tools.auth import ModisSession
from modis_tools.resources import CollectionApi, GranuleApi
from modis_tools.granule_handler import GranuleHandler

username = "guo_zhengfei@163.com"  # Update this line
password = "Zf011700"  # Update this line
# Authenticate a session
session = ModisSession(username=username, password=password)

# Query the MODIS catalog for collections
collection_client = CollectionApi(session=session)
collections = collection_client.query(short_name="MCD19A3D", version="061")
granule_client = GranuleApi.from_collection(collections[0], session=session)

nigeria_bbox = [2.1448863675, 4.002583177, 15.289420717, 14.275061098]
nigeria_granules = granule_client.query(start_date="2016-01-01", end_date="2018-12-31", bounding_box=nigeria_bbox)

GranuleHandler.download_from_granules(nigeria_granules, session)

