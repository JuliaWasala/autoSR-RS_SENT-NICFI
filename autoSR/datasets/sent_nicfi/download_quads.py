import os
import urllib.request

import requests


def get_download_links(session, api_url, mosaic, parameters, quad_ids):
    """Retrieves download links for specified quads
    inputs:
        session: requests.Session
        api_url: string         api download url
        mosaic: string          json object corresponding to desired
                                mosaic (e.g. planet_medres_visual_2021-06_mosaic)
        parameters: dict        api request parameters
        quad ids:   list(str)   list of quad_ids
    returns:
        download_links: list(str) download links for each of the quads
    """
    download_links = []
    for id in quad_ids:
        # make get request to access mosaic from basemaps API
        res = session.get(
            f"{api_url}/{mosaic['mosaics'][0]['id']}/quads/{id}", params=parameters
        )
        download_links.append(res.json()["_links"]["download"])
    return download_links


def download_quads(download_links, dest_dir, quad_ids):
    """Download quads from planet api
    inputs:
        download_links: list(str)       list of download links for the quads
        dest_dir: str                   path where quads have to be saved
        quad_ids: list(str)             list of quad ids
    """
    for i, link in enumerate(download_links):
        name = quad_ids[i] + ".tif"
        filename = os.path.join(dest_dir, name)

        if not os.path.isfile(filename):
            urllib.request.urlretrieve(link, filename)


quad_ids = [
    "1183-869",
    "1269-1090",
    "1200-920",
    "978-1096",
    "1137-956",
    "1075-1128",
    "1304-1079",
    "1110-882",
    "1208-1129",
    "1032-1107",
    "1134-1020",
    "1102-1028",
    "1013-1053",
    "1304-935",
    "1176-1016",
    "1253-938",
    "943-1098",
    "1056-1076",
    "1171-946",
    "1282-1037",
    "1235-1016",
    "1203-977",
    "1288-931",
    "1245-1109",
    "1109-944",
    "1149-1085",
    "1019-1090",
    "982-1096",
    "1188-895",
    "1005-1089",
]


if __name__ == "__main__":
    try:
        PLANET_API_KEY = os.environ["PLANET_API_KEY"]
    except KeyError as e:
        raise Exception("PLANET_API_KEY environment variable not exported") from e

    API_URL = "https://api.planet.com/basemaps/v1/mosaics"
    session = requests.Session()
    session.auth = (PLANET_API_KEY, "")

    parameters = {
        "name__is": "planet_medres_visual_2021-06_mosaic"  # <= customize to your use case
    }
    res = session.get(API_URL, params=parameters)
    mosaic = res.json()

    download_quads(
        get_download_links(session, API_URL, mosaic, parameters, quad_ids),
        "/data/s1620444/automl/datasets/sent_nicfi/nicfi/original",
        quad_ids,
    )
