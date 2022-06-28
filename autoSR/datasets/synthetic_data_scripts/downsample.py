import os
from os import listdir
from os.path import isfile, join

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling


def downsample(folder, name, filetype="tif", factor=2, alg=Image.BICUBIC, hr_name="hr"):
    hr_all_files = [f for f in listdir(
        folder+hr_name) if isfile(join(folder+hr_name, f))]
    # keep only tif
    hr_files = [f for f in hr_all_files if f.endswith(filetype)]
    os.makedirs(folder+"/"+name, exist_ok=True)

    for f in hr_files:
        print(f"file = {join(folder+hr_name,f)}")
        try:
            hr = Image.open(join(folder+hr_name, f))
            lr_size = (hr.size[0]//factor, hr.size[1]//factor)
            lr = hr.resize(lr_size, resample=alg)
            lr.save(join(folder+"/"+name, f), save_all=True)
        except:
            hr = rasterio.open(join(folder+hr_name, f))
            lr = hr.read(out_shape=(hr.count, int(hr.height/factor),
                         int(hr.width/factor)), resampling=Resampling.cubic)
            im = Image.fromarray(np.moveaxis(lr, 0, -1).astype("uint8"), 'RGB')
            im.save(join(folder+"/"+name, f), save_all=True)
