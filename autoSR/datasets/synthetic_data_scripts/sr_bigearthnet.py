

import os

import rasterio

from check_im_sizes import check_im_sizes
from downsample import downsample

if __name__ == "__main__":

    # DOWNLOAD ORIGINAL DATA
    # AND PLACE IN DATADIR

    DATADIR = "/data/s1620444/automl/datasets/"
    IMAGES = DATADIR+"BigEarthNet-v1.0/"
    BIGEARTHNETDIR = DATADIR+"sr_bigearthnet/"

    # make new folder for synthetic dataset and move images to hr folder
    # if not yet exists

    if not (os.path.isdir(BIGEARTHNETDIR+"train_hr") and os.path.isdir(BIGEARTHNETDIR+"val_hr")):
        print(f"creating {BIGEARTHNETDIR}hr")
        os.makedirs(BIGEARTHNETDIR, exist_ok=True)
        os.makedirs(BIGEARTHNETDIR+"hr", exist_ok=True)

    # merge bands and write
    for subdir, _, _ in os.walk(IMAGES):
        if subdir != IMAGES:
            print(f"Processing scene {subdir}...")
            name = os.path.basename(subdir)
            print(name)
            r_band = rasterio.open(os.path.join(
                subdir, name+"_B04.tif"))
            b_band = rasterio.open(os.path.join(
                subdir, name+"_B02.tif"))
            g_band = rasterio.open(os.path.join(
                subdir, name+"_B03.tif"))

            band_geo = r_band.profile
            band_geo.update({"count": 3})

            with rasterio.open(os.path.join(BIGEARTHNETDIR, "hr", name+".tif"), 'w', **band_geo) as dest:
                dest.write(r_band.read(1), 1)
                dest.write(b_band.read(1), 2)
                dest.write(g_band.read(1), 3)

    # # removing files with wrong sizes
    print("removing images with wrong size")
    check_im_sizes(BIGEARTHNETDIR+'hr/', (120, 120))

    print("Creating lr_bicubic_2x")
    downsample(BIGEARTHNETDIR, "lr_bicubic_2x")

    print("Done")
