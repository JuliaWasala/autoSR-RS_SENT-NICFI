This folder contains scripts to export the datasets used in this research
as tensorflow datasets.

# before running
In most cases a corresponding script must be run first.
* For cerrado, sr_ucmerced, sr_so2sat these are found in ../synthethic_data_scripts
* The description & code on how to construct sent_nicfi is in ../sent_nicfi
* OLI2MSI can be downloaded and then the corresponding tfds script ran right away


# running the scripts
The scripts can be run with:
$ tfds build --data_dir='/data/s1620444/automl/datasets/tfds' --manual_dir='/data/s1620444/automl/datasets' <dataset_name>.py

use the automl-sr-rs-env

Replace the paths with your own path. data_dir is where the compiled tfds dataset is stored, manual_dir where the source images are. 