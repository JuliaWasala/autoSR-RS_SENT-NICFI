from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tiffile as tiff

_CITATION = r"""
        @article{Sumbul2019BigEarthNetAL,
  title={BigEarthNet: A Large-Scale Benchmark Archive For Remote Sensing Image Understanding},
  author={Gencer Sumbul and Marcela Charfuelan and Beg{"u}m Demir and Volker Markl},
  journal={CoRR},
  year={2019},
  volume={abs/1902.06148}
}
    
        """


class sr_bigearthnet(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for BigEarthNet-v1.0."""
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download the v.1 file from bigearth.net
    homepages.dcc.ufmg.br uses an invalid security certificate. Run ../synthetic_data_scripts/sr_bigearthnet.py to get the synthetic data.
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
        Sentinel 2 RGB dataset
        """,
            features=tfds.features.FeaturesDict({
                'lr': tfds.features.Tensor(shape=(60, 60, 3), dtype=tf.uint8),
                'hr': tfds.features.Tensor(shape=(120, 120, 3), dtype=tf.uint8),
            }),
            supervised_keys=('lr', 'hr'),
            homepage='http://bigearth.net/',
            # Bibtex citation for the dataset
            citation=_CITATION,
        )

    def _generate_examples(self, lr_path, hr_path):
        """Yields examples."""
        for root, _, files in tf.io.gfile.walk(lr_path):
            for file_path in files:

                lr = tiff.imread(Path(lr_path)/file_path)
                hr = tiff.imread(Path(hr_path)/file_path)

                # Select only tif files.
                if file_path.endswith(".tif"):
                    yield file_path, {
                        "lr": lr.astype("uint8"),
                        "hr": np.moveaxis(hr, 0, -1).astype("uint8"),
                    }

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        extracted_path = dl_manager.manual_dir/'sr_bigearthnet'
        return {"train": self._generate_examples(hr_path=extracted_path/'hr', lr_path=extracted_path/'lr_bicubic_2x')}
