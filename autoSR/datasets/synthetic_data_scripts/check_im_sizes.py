import os

import rasterio
from PIL import Image


def check_im_sizes(hr_path, required_size):
    num_wrongsize = 0
    for _, _, file in os.walk(hr_path):
        for f in file:
            try:
                im = Image.open(hr_path+f)
                if im.size != required_size:
                    num_wrongsize += 1
                    os.remove(hr_path+f)
            except:
                im = rasterio.open(hr_path+f)
                if (im.width, im.height) != required_size:
                    num_wrongsize += 1
                    os.remove(hr_path+f)
