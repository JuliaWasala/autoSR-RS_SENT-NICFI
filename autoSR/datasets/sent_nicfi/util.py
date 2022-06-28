import os
import re
from typing import List

from shapely.geometry import Polygon


def write_gdalinfo(dir, tif_name):
    os.system(
        f'gdalinfo {os.path.join(dir, tif_name)} > {os.path.join(dir,os.path.splitext(tif_name)[0])}_info.txt')


def extract_coords(line: str) -> str:
    regex = r'[0-9]+d[ ]?[0-9]+\'[ ]?[0-9]+\.[0-9]+"[EW],[ ]+[0-9]+d[ ]?[0-9]+\'[ ]?[0-9]+\.[0-9]+"[SN]'
    match = re.findall(regex, line)[0].split(",")
    return [s.replace(" ", "") for s in match]

# https://stackoverflow.com/questions/33997361/how-to-convert-degree-minute-second-to-degree-decimal


def ddm2dec(dms):
    """Return decimal representation of DDM (degree decimal minutes)
    input format 45d17'96"N
    """
    deg, minutes, seconds, direction = re.split('[d\'"]', dms)
    return (float(deg) + float(minutes)/60 + float(seconds)/(60*60)) * (-1 if direction in ['W', 'S'] else 1)


def get_dec_coordinates(line):
    dms_coords = extract_coords(line)
    return tuple(ddm2dec(coord) for coord in dms_coords)


def get_coordinates(filepath: str) -> List:
    with open(filepath) as f:
        lines = f.readlines()
        print(lines)
        coordinates = [[line for line in lines if "Upper Left" in line],
                       [line for line in lines if "Upper Right" in line],
                       [line for line in lines if "Lower Right" in line],
                       [line for line in lines if "Lower Left" in line]
                       ]
    print(coordinates)
    return [get_dec_coordinates(line[0]) for line in coordinates]


def get_nicfi_polygons(nicfi_dir):
    nicfi_tile_names = []
    for file in os.listdir(nicfi_dir):
        if file.endswith(".tif"):
            nicfi_tile_names.append(file)
            if not os.path.exists(os.path.join(nicfi_dir, os.path.splitext(file)[0])+"_info.txt"):
                write_gdalinfo(nicfi_dir, file)

    # get polygons
    nicfi_areas = {}
    print(nicfi_tile_names)
    for tile in nicfi_tile_names:

        gdalinfo_file = os.path.join(
            nicfi_dir, os.path.splitext(tile)[0])+"_info.txt"
        print("file")
        print(gdalinfo_file)
        nicfi_areas[tile] = Polygon(get_coordinates(gdalinfo_file))

    return nicfi_areas


def get_polygon(dir, tilename):
    if not os.path.exists(os.path.join(dir, os.path.splitext(tilename)[0])+"_info.txt"):
        write_gdalinfo(dir, tilename)

    gdalinfo_file = os.path.join(
        dir, os.path.splitext(tilename)[0])+"_info.txt"
    return Polygon(get_coordinates(gdalinfo_file))
