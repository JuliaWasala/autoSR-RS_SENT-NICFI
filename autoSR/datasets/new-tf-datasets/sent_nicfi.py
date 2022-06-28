from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tiffile as tiff

CITIES = ["1183-869", "1269-1090", "1200-920", "978-1096", "1137-956"]
DESERT = ["1075-1128", "1304-1079", "1110-882", "1208-1129", "1032-1107"]
FOREST = ["1134-1020", "1102-1028", "1013-1053", "1304-935", "1176-1016"]
AGRICULTURE = ["1253-938", "943-1098", "1056-1076", "1171-946", "1282-1037"]
SAVANNA = ["1149-1085", "1019-1090", "982-1096", "1188-895", "1005-1089"]
MISC = ["1235-1016", "1203-977", "1288-931", "1245-1109", "1128-953"]

class sent_nicfi(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for SENT-NICFI dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    For instructions see https://www.github.com/JuliaWasala/automl-sr-rs
    in folder: autoSR/datasets/sent_nicfi"""
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="""
        SENT-NICFI is a multisensor dataset for remote sensing imagery super-resolution using 
        Sentinel-2 images and Planetscope images from the NICFI project. The dataset can be split according
        to ecosystem: cities, desert, forest, agriculture, miscellaneous and savanna.
        """,
            features=tfds.features.FeaturesDict(
                {
                    "lr": tfds.features.Tensor(shape=(100, 100, 3), dtype=tf.uint8),
                    "hr": tfds.features.Tensor(shape=(200, 200, 3), dtype=tf.uint8),
                }
            ),
            supervised_keys=("lr", "hr"),
            # Bibtex citation for the dataset
        )

    @staticmethod
    def _check_ecosystem(filename, ecosystem):
        """Checks whether filename corresponds to specified ecosystem
        inputs:
            filename: str of format <INT>-<INT>_<INT>_<INT>.tif
            ecosystem: None |           None if any ecosystem.
                        CITIES |
                        DESERT |
                        FOREST |
                        AGRICULTURE |
                        MISC |
                        SAVANNA
        """
        if ecosystem is None:
            return True
        return any(id in filename for id in ecosystem)

    def _generate_examples(self, lr_path, hr_path, ecosystem=None):
        """Yields examples."""
        # walks over LR, ignoring the hr file I havent downloaded yet
        for root, _, files in tf.io.gfile.walk(lr_path):
            for file_path in files:
                # Select only tif files.
                if file_path.endswith(".tif") and self._check_ecosystem(
                    file_path, ecosystem
                ):

                    yield file_path, {
                        "lr": np.array(tiff.imread(Path(lr_path) / file_path)),
                        "hr": np.array(tiff.imread(Path(hr_path) / file_path)),
                    }

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        extracted_path = dl_manager.manual_dir / "sent_nicfi"
        return {
            "train": self._generate_examples(
                hr_path=extracted_path / "nicfi" / "5m_color_corrected_splits",
                lr_path=extracted_path / "sent" / "10m_splits",
            ),
            "cities": self._generate_examples(
                hr_path=extracted_path / "nicfi" / "5m_color_corrected_splits",
                lr_path=extracted_path / "sent" / "10m_splits",
                ecosystem=CITIES,
            ),
            "desert": self._generate_examples(
                hr_path=extracted_path / "nicfi" / "5m_color_corrected_splits",
                lr_path=extracted_path / "sent" / "10m_splits",
                ecosystem=DESERT,
            ),
            "forest": self._generate_examples(
                hr_path=extracted_path / "nicfi" / "5m_color_corrected_splits",
                lr_path=extracted_path / "sent" / "10m_splits",
                ecosystem=FOREST,
            ),
            "agriculture": self._generate_examples(
                hr_path=extracted_path / "nicfi" / "5m_color_corrected_splits",
                lr_path=extracted_path / "sent" / "10m_splits",
                ecosystem=AGRICULTURE,
            ),
            "misc": self._generate_examples(
                hr_path=extracted_path / "nicfi" / "5m_color_corrected_splits",
                lr_path=extracted_path / "sent" / "10m_splits",
                ecosystem=MISC,
            ),
            "savanna": self._generate_examples(
                hr_path=extracted_path / "nicfi" / "5m_color_corrected_splits",
                lr_path=extracted_path / "sent" / "10m_splits",
                ecosystem=SAVANNA,
            ),
        }
