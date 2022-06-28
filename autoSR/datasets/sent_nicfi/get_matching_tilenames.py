import hashlib
import os
from datetime import datetime
from argparse import ArgumentParser
from typing import List, Tuple

import pandas as pd
from sentinelloader import Sentinel2Loader
from sentinelsat import SentinelAPI

from util import get_polygon
from shapely.geometry import Polygon
import rasterio


def get_offline_tiles_uuids(datadir: str, tile_hash: str) -> pd.DataFrame:
    """ Returns a list of offline tile uuids
    inputs:
        datadr: str    file path of dir with files
        tile_hash: str  hash of area
    returns:
        list of uuids"""
    uuids=[]
    for file in os.listdir(datadir):
        if file.endswith("csv") and tile_hash in file:
            df = pd.read_csv(os.path.join(datadir,file))
            if not df.empty:
                uuids.extend(df["uuid"].to_list())
    return uuids

def get_matching_tilenames(sl, polygon, start_date,end_date, datadir="/data/s1620444/automl/datasets/sent_nicfi/sent/apiquery", removeCache = True) -> Tuple[List, List]:
    online_files = sl.getRegionHistory(polygon, "TCI", "10m", start_date, end_date)

    bbox = rasterio.features.bounds(polygon)
    geoPolygon = [(bbox[0], bbox[3]), (bbox[0], bbox[1]),
        (bbox[2], bbox[1]), (bbox[2], bbox[3])]

    area = Polygon(geoPolygon).wkt
    area_hash = hashlib.md5(area.encode()).hexdigest()
    print(f"area hash: {area_hash}")

    offline_files = get_offline_tiles_uuids(datadir, area_hash)
    return online_files, list(set(offline_files))

def trigger_loading_offline_tiles(api, offline_tiles):
    unsuccessful_downloads = []
    for id in offline_tiles:
        try:
            api.trigger_offline_retrieval(id)
        except:
            print(f"Download of {id} unsuccessful")
            unsuccessful_downloads.append(id)
    return unsuccessful_downloads

if __name__=="__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--tilename", type=str)
    argparser.add_argument("--startdate", type=str,default="06-01")
    argparser.add_argument("--enddate", type=str,default="06-10")

    args = argparser.parse_args()
    data_dir= "/data/s1620444/automl/datasets/sent_nicfi"
    sent_dir = os.path.join(data_dir, "sent")
    nicfi_dir = os.path.join(data_dir, "nicfi/original")
    year = "2021"
    start_date = f"{year}-{args.startdate}"
    end_date = f"{year}-{args.enddate}"

    # initialize connection
    sl = Sentinel2Loader(sent_dir, os.environ["COPERNICUS_USER"], os.environ["COPERNICUS_PASSWORD"],
                        apiUrl='https://scihub.copernicus.eu/dhus', showProgressbars=True,cloudCoverage=(0,10))
    api = SentinelAPI(os.environ["COPERNICUS_USER"], os.environ["COPERNICUS_PASSWORD"], 'https://scihub.copernicus.eu/dhus')

    area = get_polygon(nicfi_dir, args.tilename)

    # get matching sentinel tiles
    online_files, offline_files= get_matching_tilenames(sl, area, start_date, end_date, removeCache=False)

    # trigger offline retrieval
    unsuccesful_downloads = trigger_loading_offline_tiles(api, offline_files)

    #remove unsuccesful downloads from offline_files
    triggered_offline_files=[f for f in offline_files if f not in unsuccesful_downloads]

    # write 
    matched_tiles_dir = os.path.splitext(os.path.join(sent_dir,args.tilename))[0]
    os.makedirs(os.path.splitext(os.path.join(sent_dir,args.tilename))[0], exist_ok=True)
    with open(os.path.join(matched_tiles_dir,"online_files.txt"),"w") as f:
        for file in online_files:
            f.write(file+"\n")
        
    with open(os.path.join(matched_tiles_dir,"offline_files.txt"),"w") as f:
        for file in triggered_offline_files:
            f.write(file+"\n")
        