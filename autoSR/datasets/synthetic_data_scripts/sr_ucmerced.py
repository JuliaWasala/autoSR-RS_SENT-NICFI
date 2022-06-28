# http://weegee.vision.ucmerced.edu/datasets/landuse.html

import os
from distutils.dir_util import copy_tree

from check_im_sizes import check_im_sizes
from downsample import downsample

if __name__ == "__main__":

    # DOWNLOAD ORIGINAL DATA FROM http://weegee.vision.ucmerced.edu/datasets/landuse.html
    # AND PLACE IN DATADIR

    DATADIR = "/data/s1620444/automl/datasets/"
    IMAGES = "/data/s1620444/automl/datasets/UCMerced_LandUse/Images/"
    UCMERCEDDIR = DATADIR+"sr_ucmerced/"

    # make new folder for synthetic dataset and move images to hr folder
    # if not yet exists
    if not os.path.isdir(UCMERCEDDIR+"hr"):
        print(f"creating {UCMERCEDDIR}hr")
        os.makedirs(UCMERCEDDIR, exist_ok=True)
        os.makedirs(UCMERCEDDIR+"hr", exist_ok=True)

        for subdir, _, _ in os.walk(IMAGES):
            if subdir != IMAGES:
                print(f"copying {subdir} to {UCMERCEDDIR+'hr/'}")
                copy_tree(subdir, UCMERCEDDIR+"hr/")

    # removing files with wrong sizes
    print("removing images with wrong size")
    check_im_sizes(UCMERCEDDIR+'hr/', (256, 256))

    # bicubic downsampling x2
    print("Creating lr_bicubic_2x")
    downsample(UCMERCEDDIR, "lr_bicubic_2x")

    print("Done")
