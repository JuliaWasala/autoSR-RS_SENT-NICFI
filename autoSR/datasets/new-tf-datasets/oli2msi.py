from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tiffile as tiff

_CITATION = r"""
            @article{wang2021multisensor,
            title={Multisensor Remote Sensing Imagery Super-Resolution with Conditional GAN},
            author={Wang, Junwei and Gao, Kun and Zhang, Zhenzhou and Ni, Chong and Hu, Zibo and Chen, Dayu and Wu, Qiong},
            journal={Journal of Remote Sensing},
            volume={2021},
            year={2021},
            publisher={AAAS}
            }
        """


class oli2msi(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for OLI2MSI dataset."""
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download the (tif) dataset from https://github.com/wjwjww/OLI2MSI, either by using the scripts or from the provided google drive link
    Place the directory in the `manual_dir/`"""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        "1.0.0": "Initial release.", }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return tfds.core.DatasetInfo(
            builder=self,
            # Description and homepage used for documentation
            description="""
        OLI2MSI is a multisensor dataset for remote sensing imagery super-resolution. 
        The OLI2MSI dataset is composed of Landsat8-OLI and Sentinel2-MSI images, where OLI images serve as low-resolution 
        (LR) images and MSI images are regarded as ground truth high-resolution (HR) images. The OLI and MSI data have 10m 
        and 30m ground sample distance (GSD) respectively, which means that the dataset has an upscale factor of 3. 
        More details about OLI2MSI can be found in our paper.
        """,
            features=tfds.features.FeaturesDict({
                'lr': tfds.features.Tensor(shape=(160, 160, 3), dtype=tf.uint8),
                'hr': tfds.features.Tensor(shape=(480, 480, 3), dtype=tf.uint8),
            }),
            supervised_keys=('lr', 'hr'),
            homepage='https://github.com/wjwjww/OLI2MSI',
            # Bibtex citation for the dataset
            citation=_CITATION,
        )

    def _generate_examples(self, lr_path, hr_path):
        """Yields examples."""
        for root, _, files in tf.io.gfile.walk(lr_path):
            for file_path in files:
                # Select only tif files.
                if file_path.endswith(".TIF"):
                    # change to channels last
                    lr_channels_first = np.array(
                        tiff.imread(Path(lr_path)/file_path))
                    hr_channels_first = np.array(
                        tiff.imread(Path(hr_path)/file_path))

                    yield file_path, {
                        "lr": (255*np.moveaxis(lr_channels_first, 0, -1)).astype("uint8").tolist(),
                        "hr": (255*np.moveaxis(hr_channels_first, 0, -1)).astype("uint8").tolist(),
                    }

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        extracted_path = dl_manager.manual_dir/'OLI2MSI'
        return {"train": self._generate_examples(hr_path=extracted_path/'train_hr', lr_path=extracted_path/'train_lr'),
                "test": self._generate_examples(hr_path=extracted_path/'test_hr', lr_path=extracted_path/'test_lr')}
