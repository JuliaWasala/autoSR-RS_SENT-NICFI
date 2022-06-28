import os

import numpy as np
from osgeo import gdal
from PIL import Image
from skimage.exposure import match_histograms
from tqdm import tqdm


def resample(source_dir, dest_dir, tilename_list, resolution):
    """Resamples tiffs in a directory to desired resolution, writes them
    to destination dir
    inputs:
        source_dir: str     path to input tiffs
        dest_dir: str       path to destination dir
        tilename_list: list(str)    list of tilenames to be resampled
        resolution: float   desired resolution (in m)"""

    kwargs = dict(xRes=resolution, yRes=resolution, resampleAlg="cubic")

    for tile in tqdm(tilename_list):
        ds = gdal.Warp(
            os.path.join(dest_dir, tile), os.path.join(source_dir, tile), **kwargs
        )
        ds = None


def split(source_dir, dest_dir, tilename_list, size):
    """Splits tiffs in dir into smaller tiles and writes to destination dir
    inputs:
        source_dir: str         path to source directory
        dest_dir: str           path to destination directory
        tilename_list: list(str)    list of tilenames to be split
        size: int               output size in px of tiles to be written
    """
    # based on https://gis.stackexchange.com/questions/221671/splitting-tif-image-into-several-tiles
    for tile in tqdm(tilename_list):
        ds = gdal.Open(os.path.join(source_dir, tile))
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize

        for i in range(0, xsize, size):
            for j in range(0, ysize, size):
                com_string = (
                    "gdal_translate -of GTIFF -srcwin "
                    + str(i)
                    + ", "
                    + str(j)
                    + ", "
                    + str(size)
                    + ", "
                    + str(size)
                    + " "
                    + os.path.join(source_dir, tile)
                    + " "
                    + os.path.join(dest_dir, os.path.splitext(tile)[0])
                    + "_"
                    + str(i // size)
                    + "_"
                    + str(j // size)
                    + ".tif"
                )
                os.system(com_string)


def resize_and_split(source, resolution, size, residual_px):
    """Resizes and splits input images and writes them
    inputs:
        source: sent | nicfi        which source images to process
        resolution: int             desired resolution
        size: int                   desired size in px of resulting tiles
        residual_px: int            number of pixels that will be left after
                                    cropping complete tiles
    """
    source_dir = f"/data/s1620444/automl/datasets/sent_nicfi/{source}/original"
    resample_dir = f"/data/s1620444/automl/datasets/sent_nicfi/{source}/{resolution}m"
    split_tiles_dir = (
        f"/data/s1620444/automl/datasets/sent_nicfi/{source}/{resolution}m_splits"
    )

    os.makedirs(resample_dir, exist_ok=True)
    os.makedirs(split_tiles_dir, exist_ok=True)

    tilenames = []
    for file in os.listdir(source_dir):
        if file.endswith(".tif"):
            tilenames.append(file)

    resample(source_dir, resample_dir, tilenames, resolution)

    print(f"Divide resampled files into tiles of size {size}x{size}px")
    split(resample_dir, split_tiles_dir, tilenames, size)

    print("Removing all incomplete tiles...")
    for file in os.listdir(split_tiles_dir):
        if residual_px in file:
            os.remove(os.path.join(split_tiles_dir, file))


def histogram_matching():
    """Performs histogram matching on NICFI images based on SENT,
    saves color corrected image"""

    sent_dir = "/data/s1620444/automl/datasets/sent_nicfi/sent/10m_splits"
    nicfi_dir = "/data/s1620444/automl/datasets/sent_nicfi/nicfi/5m_splits"
    nicfi_color_corrected_dir = (
        "/data/s1620444/automl/datasets/sent_nicfi/nicfi/5m_color_corrected_splits"
    )
    os.makedirs(nicfi_color_corrected_dir, exist_ok=True)

    for file in os.listdir(sent_dir):
        if file.endswith("tif"):
            sent_img = np.array(gdal.Open(f"{sent_dir}/{file}").ReadAsArray())

            nicfi_img = np.array(gdal.Open(f"{nicfi_dir}/{file}").ReadAsArray())[
                :3, :, :
            ]

            color_corrected_nicfi = Image.fromarray(
                np.moveaxis(
                    match_histograms(nicfi_img, sent_img, channel_axis=0), 0, -1
                )
            )
            color_corrected_nicfi.save(
                f"{nicfi_color_corrected_dir}/{file}", save_all=True
            )


if __name__ == "__main__":

    for source in ["sent", "nicfi"]:
        if source == "sent":
            RESOLUTION = 10  # m
            SIZE = 100  # px
            RESIDUAL_PX = "1900"  # px
        elif source == "nicfi":
            RESOLUTION = 5  # m
            SIZE = 200  # px
            RESIDUAL_PX = "3800"  # px

        resize_and_split(source, RESOLUTION, SIZE, RESIDUAL_PX)
        histogram_matching()
