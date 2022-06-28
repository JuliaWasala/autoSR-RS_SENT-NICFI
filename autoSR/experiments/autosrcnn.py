import os
import time
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from auto_models.models.autosrcnn import autoSRCNN
from tensorflow import keras
from utils import get_data, psnr, ssim

DATADIR = os.environ["DATADIR"]
DATASETSDIR = os.environ["DATASETSDIR"]
RESULTSDIR = os.environ["RESULTSDIR"]

if __name__ == "__main__":
    start_t = time.time()
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        choices=['cerrado', 'sr_ucmerced', 'oli2msi', 'sr_so2sat','sent_nicfi'])
    parser.add_argument("--trials", type=int)
    parser.add_argument("--id", type=int)
    args = parser.parse_args()

    if args.dataset == "oli2msi":
        scale = 3
    else:
        scale = 2

    # get data
    # check whether has to be reshaped to numpy
    test_ds, valid_ds, train_ds = get_data(args.dataset,
                                           train_batch_size=-1,
                                           test_batch_size=-1,
                                           val_batch_size=-1,
                                           as_supervised=False)

    train_ds_arr = tfds.as_numpy(train_ds)
    test_ds_arr = tfds.as_numpy(test_ds)
    valid_ds_arr = tfds.as_numpy(valid_ds)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"{RESULTSDIR}/tensorboard/autosrcnn")

    auto_model = autoSRCNN(os.path.join(
        RESULTSDIR, "autosrcnn"), f"{args.id}_autosrcnn_{args.dataset}", max_trials=args.trials, scale=scale)
    auto_model.fit(x=train_ds_arr["lr"], y=train_ds_arr["hr"], callbacks=[tensorboard_callback, tf.keras.callbacks.EarlyStopping(
        patience=10, min_delta=1e-4)], epochs=100, validation_data=(valid_ds_arr["lr"], valid_ds_arr["hr"]), verbose=2)
    results = np.asarray([auto_model.evaluate(
        test_ds_arr["lr"], test_ds_arr["hr"], custom_objects={"psnr": psnr, "ssim": ssim})])

    print(results)
    with open(RESULTSDIR+"/autosrcnn/autosrcnn.csv", "a") as f:
        f.write(
            f"{args.trials},{args.dataset},{','.join([str(i) for i in results[0]])}\n")
    best_model = auto_model.export_model(
        custom_objects={"psnr": psnr, "ssim": ssim})
    print(best_model.summary())
    best_model.save(
        RESULTSDIR+f"/weights/autosrcnn_{args.dataset}_{args.trials}_{args.id}.h5")
    end = time.time()
    hours, rem = divmod(end-start_t, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
