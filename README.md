## AutoSR-RS_SENT-NICFI

This repository contains the code for AutoSR-RS: a NAS method for super resolution of remote sensing images.
The method uses pretrained weights that are available at: https://universiteitleiden.data.surfsara.nl/index.php/s/i9oF0o8R36ZXZ5X.

The datasets used are available at : https://universiteitleiden.data.surfsara.nl/index.php/s/MBZs6OyPOiELRNZ.
For instructions on how to re-create SENT-NICFI, look at the autoSR/datasets/sent-nicfi folder.

# Code folders (in autoSR)

- auto_models:
  custom blocks & autokeras automodels

- baselines:
  adapted baselines + code for training + eval

- datasets:
  code for creating datasets

- experiments:
  code for experiments with auto_models

- results_processing:
  all the code for processing results and producing plots.

# Requirements

More information on the requirements is in the env_documentation folder

To run scripts, environment variables have to be set. (Note: no need to do this for the jupyter  
notebooks: variables are exported there in the notebook itself).  
Update config.sh file with your paths & run before running scripts:

- RESULTSDIR is where training results, model checkpoints and weights are saved
- DATADIR is where "raw" data is stored that is used to create TFDS datasets
- DATASETSDIR is dir where compiled TFDS datasets live
- PROJECTDIR is the path to the code folder
