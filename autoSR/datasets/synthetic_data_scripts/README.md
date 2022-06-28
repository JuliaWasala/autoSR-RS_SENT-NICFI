This folder contains code to create synthethic SISR datasets
from remote sensing images originally from classification datasets.

# datasets
- Brazilian Cerrado Savanna -> cerrado
- UC Merced -> sr_ucmerced
- So2Sat -> sr_so2sat

The original images are downsampled 2x in order to create synthethic datasets. In some cases not all images in the dataset have the exact same size, in this case the deviant images are removed with the help of check_im_sizes.py.
