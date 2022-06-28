## AutoSR-RS_SENT-NICFI

This repository contains the code for AutoSR-RS: a NAS method for super resolution of remote sensing images.
The method requires using pretrained weights that are available at: https://universiteitleiden.data.surfsara.nl/index.php/s/i9oF0o8R36ZXZ5X.

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
  

# Environment Requirements
NOTE: Do not forget to download the pretrained weights, these are REQUIRED: https://universiteitleiden.data.surfsara.nl/index.php/s/i9oF0o8R36ZXZ5X.
More information on the environment requirements is in the env_documentation folder

To run scripts, environment variables have to be set. (Note: no need to do this for the jupyter  
notebooks: variables are exported there in the notebook itself).  
Update config.sh file with your paths & run before running scripts:

- RESULTSDIR is where training results, model checkpoints and weights are saved
- DATADIR is where "raw" data is stored that is used to create TFDS datasets
- DATASETSDIR is dir where compiled TFDS datasets live
- PROJECTDIR is the path to the code folder

# Example
For example code, see autoSR/experiments/autosr.py:

Imports:
```
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from auto_models.models.autosr_rs import autoSR_RS
from utils import get_data, psnr, ssim

DATADIR = os.environ["DATADIR"]
DATASETSDIR = os.environ["DATASETSDIR"]
RESULTSDIR = os.environ["RESULTSDIR"]
```

Load data:
```
test_ds, valid_ds, train_ds = get_data("sent_nicfi",
                                       train_batch_size=-1,
                                       test_batch_size=-1,
                                       val_batch_size=-1,
                                       as_supervised=False)

train_ds_arr = tfds.as_numpy(train_ds)
test_ds_arr = tfds.as_numpy(test_ds)
valid_ds_arr = tfds.as_numpy(valid_ds)
```

Set pretrained weights choices:
```
dataset_options=['cerrado', 'sr_ucmerced', 'oli2msi', 'sr_so2sat']
```

Load autoSR_RS:
```
auto_model = autoSR_RS(dataset_options,os.path.join(RESULTSDIR, "autosr","v1"),
                       f"autosr_cerrado", "1",max_trials=20, scale=2,overwrite=True)
```

Fit the model:
```
auto_model.fit(x=train_ds_arr["lr"], y=train_ds_arr["hr"], callbacks=[tf.keras.callbacks.EarlyStopping(
    patience=10, min_delta=1e-4)], epochs=100, validation_data=(valid_ds_arr["lr"], valid_ds_arr["hr"]), verbose=2)
```

Get test results:
```
results = np.asarray([auto_model.evaluate(
    test_ds_arr["lr"], test_ds_arr["hr"], custom_objects={"psnr": psnr, "ssim": ssim})])
```

Save the best model (See AutoKeras documentation for more information about how models are saved https://autokeras.com/tutorial/export/)
```
best_model = auto_model.export_model(
    custom_objects={"psnr": psnr, "ssim": ssim})
print(best_model.summary())
tf.saved_model.save(best_model,
    RESULTSDIR+f"/weights/autosr_cerrado.h5")
```
