from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
import tiffile as tiff

_CITATION = r"""
@inproceedings{yang2010bag,
  title={Bag-of-visual-words and spatial extensions for land-use classification},
  author={Yang, Yi and Newsam, Shawn},
  booktitle={Proceedings of the 18th SIGSPATIAL international conference on advances in geographic information systems},
  pages={270--279},
  year={2010}
}
        """

class sr_ucmerced(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for UC Merced Dataset."""
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download the data from http://weegee.vision.ucmerced.edu/datasets/landuse.html
    Run ../synthetic_data_scripts/sr_ucmerced.py to get the synthetic data.
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
        Synthetic dataset based on Brazildam.
        Brazildam dataset consists of multispectral images of ore tailings dams throughout Brazil
        """,
            features=tfds.features.FeaturesDict({
                'lr': tfds.features.Tensor(shape=(128, 128, 3), dtype=tf.uint8),
                'hr': tfds.features.Tensor(shape=(256, 256, 3), dtype=tf.uint8),
            }),
            supervised_keys=('lr', 'hr'),
            homepage='http://weegee.vision.ucmerced.edu/datasets/landuse.html',
            # Bibtex citation for the dataset
            citation=_CITATION,
        )

    def _generate_examples(self, lr_path, hr_path):
        """Yields examples."""
        for root, _, files in tf.io.gfile.walk(lr_path):
            for file_path in files:
                # lr images have different range:
                # extract the row & col num and multiply by 2

                if file_path.endswith(".tif"):
                    yield file_path, {
                        "lr": tiff.imread(Path(lr_path)/file_path),
                        "hr": tiff.imread(Path(hr_path)/file_path),
                    }

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        extracted_path = dl_manager.manual_dir/'sr_ucmerced'
        return {"train": self._generate_examples(hr_path=extracted_path/'hr', lr_path=extracted_path/'lr_bicubic_2x')}
