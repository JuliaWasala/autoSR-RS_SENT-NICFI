# http://weegee.vision.ucmerced.edu/datasets/landuse.html

import os

import h5py
import numpy as np
from PIL import Image

from downsample import downsample

# Calibration for the optical RGB channels of Sentinel-2 in this dataset.
_OPTICAL_CALIBRATION_FACTOR = 3.5 * 255.0


def _create_rgb(sen2_bands):
    return np.clip(sen2_bands[..., [2, 1, 0]] * _OPTICAL_CALIBRATION_FACTOR, 0,
                   255).astype(np.uint8)


if __name__ == "__main__":

    # DOWNLOAD ORIGINAL DATA FROM http://weegee.vision.ucmerced.edu/datasets/landuse.html
    # AND PLACE IN DATADIR

    DATADIR = "/data/s1620444/automl/datasets/"
    IMAGES = DATADIR+"So2Sat/"
    SO2SATDIR = DATADIR+"sr_so2sat/"

    # make new folder for synthetic dataset and move images to hr folder
    # if not yet exists

    if not (os.path.isdir(SO2SATDIR+"train_hr") and os.path.isdir(SO2SATDIR+"val_hr")):
        print(f"creating {SO2SATDIR}hr")
        os.makedirs(SO2SATDIR, exist_ok=True)
        os.makedirs(SO2SATDIR+"train_hr", exist_ok=True)
        os.makedirs(SO2SATDIR+"val_hr", exist_ok=True)

    print("creating RGB tifs...")
    train_fid = h5py.File(IMAGES+"training.h5")
    val_fid = h5py.File(IMAGES+"validation.h5")

    print("training data....")
    s2_train = np.array(train_fid['sen2'])
    num_train = s2_train.shape[0]

    for i in range(num_train):
        im = Image.fromarray(_create_rgb(s2_train[i, :, :, :]))
        im.save(SO2SATDIR+"train_hr/"+str(i)+".tif", save_all=True)

    print("validation data.....")
    s2_val = np.array(val_fid['sen2'])
    num_val = s2_val.shape[0]

    for i in range(num_val):
        im = Image.fromarray(_create_rgb(s2_val[i, :, :, :]))
        im.save(SO2SATDIR+"val_hr/"+str(i)+".tif", save_all=True)

    # removing files with wrong sizes
    # print("removing images with wrong size")
    # check_im_sizes(SO2SATDIR+'hr/', (256,256))

    # # bicubic downsampling x2
    print("Creating train_lr_bicubic_2x")
    downsample(SO2SATDIR, "train_lr_bicubic_2x", hr_name="train_hr")

    print("Creating val_lr_bicubic_2x")
    downsample(SO2SATDIR, "val_lr_bicubic_2x", hr_name="val_hr")

    print("Done")
