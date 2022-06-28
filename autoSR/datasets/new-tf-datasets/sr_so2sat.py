# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""So2SAT remote sensing dataset."""

from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import tiffile as tiff

_DESCRIPTION = """\
The dataset is based in So2Sat images. The images are downsampled to create
a dataset for super resolution.
So2Sat LCZ42 is a dataset consisting of co-registered synthetic aperture radar
and multispectral optical image patches acquired by the Sentinel-1 and
Sentinel-2 remote sensing satellites, and the corresponding local climate zones
(LCZ) label. The dataset is distributed over 42 cities across different
continents and cultural regions of the world.
The dataset contains the 3 optical frequency bands of Sentinel-2, rescaled and encoded as JPEG.
Dataset URL: http://doi.org/10.14459/2018MP1454690
License: http://creativecommons.org/licenses/by/4.0
"""

# Calibration for the optical RGB channels of Sentinel-2 in this dataset.
_OPTICAL_CALIBRATION_FACTOR = 3.5 * 255.0


class sr_so2sat(tfds.core.GeneratorBasedBuilder):
    """So2SAT remote sensing dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Download the  from https://mediatum.ub.tum.de/1454690
  Run the ../synthetic_data_scripts/sr_so2sat.py script 
  Place the directory in the `manual_dir/`"""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        "1.0.0": "Initial release.", }

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'hr': tfds.features.Image(shape=[32, 32, 3]),
                'lr': tfds.features.Image(shape=[16, 16, 3])
            }),
            supervised_keys=("lr", "hr"),
            homepage='http://doi.org/10.14459/2018MP1454690',
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        extracted_path = dl_manager.manual_dir/'sr_so2sat'
        return {"train": self._generate_examples(hr_path=extracted_path/'train_hr', lr_path=extracted_path/'train_lr_bicubic_2x'),
                "validation": self._generate_examples(hr_path=extracted_path/'val_hr', lr_path=extracted_path/'val_lr_bicubic_2x')}

    def _generate_examples(self, lr_path, hr_path):
        """Yields examples."""
        for root, _, files in tf.io.gfile.walk(lr_path):
            for file_path in files:
                # Select only tif files.
                if file_path.endswith(".tif"):
                    yield file_path, {
                        "lr": tiff.imread(Path(lr_path)/file_path),
                        "hr": tiff.imread(Path(hr_path)/file_path),
                    }
