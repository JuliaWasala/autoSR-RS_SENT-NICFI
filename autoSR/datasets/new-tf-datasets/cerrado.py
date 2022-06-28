import os
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
import tiffile as tiff

_CITATION = r"""
        @inproceedings{nogueira2016towards,
        title={Towards vegetation species discrimination by using data-driven descriptors},
        author={Nogueira, Keiller and Dos Santos, Jefersson A and Fornazari, Tamires and Silva, Thiago Sanna Freire and Morellato, Leonor Patricia and Torres, Ricardo da S},
        booktitle={2016 9th IAPR Workshop on Pattern Recogniton in Remote Sensing (PRRS)},
        pages={1--6},
        year={2016},
        organization={Ieee}
        }
    
        """


class cerrado(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Brazilian Cerrado-Savanna Scenes Dataset."""
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download the file from https://homepages.dcc.ufmg.br/~keiller.nogueira/datasets/brazilian_cerrado_dataset.zip
    homepages.dcc.ufmg.br uses an invalid security certificate. Run ../synthetic_data_scripts/cerrado.py to get the synthetic data.
    Place the directory in the `manual_dir/` (~tensorflow_datasets/downloads/manual)"""
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
                'lr': tfds.features.Tensor(shape=(32, 32, 3), dtype=tf.uint8),
                'hr': tfds.features.Tensor(shape=(64, 64, 3), dtype=tf.uint8),
            }),
            supervised_keys=('lr', 'hr'),
            homepage='http://www.patreo.dcc.ufmg.br/2017/11/12/brazilian-cerrado-savanna-scenes-dataset/',
            # Bibtex citation for the dataset
            citation=_CITATION,
        )

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

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        extracted_path = dl_manager.manual_dir/'cerrado'
        return {"train": self._generate_examples(hr_path=extracted_path/'hr', lr_path=extracted_path/'lr_bicubic_2x')}
