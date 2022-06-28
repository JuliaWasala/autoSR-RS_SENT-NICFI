import os
import time
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from auto_models.models.autosr_rs import autoSR_RS
from utils import get_data, psnr, ssim

DATADIR = os.environ["DATADIR"]
DATASETSDIR = os.environ["DATASETSDIR"]
RESULTSDIR = os.environ["RESULTSDIR"]

if __name__ == "__main__":
    start_t = time.time()
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        choices=['cerrado', 'sr_ucmerced', 'oli2msi', 'sr_so2sat','sent_nicfi'])
    # parser.add_argument("--version", type=str, choices=['a', 'b', 'c', 'd'])
    parser.add_argument("--trials", type=int)
    parser.add_argument("--id", type=int)
    args = parser.parse_args()

    if args.dataset == "oli2msi":
        SCALE = 3
    else:
        SCALE = 2

    # get data
    test_ds, valid_ds, train_ds = get_data(args.dataset,
                                           train_batch_size=-1,
                                           test_batch_size=-1,
                                           val_batch_size=-1,
                                           as_supervised=False)

    train_ds_arr = tfds.as_numpy(train_ds)
    test_ds_arr = tfds.as_numpy(test_ds)
    valid_ds_arr = tfds.as_numpy(valid_ds)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"{RESULTSDIR}/tensorboard/autosr")

    dataset_options=['cerrado', 'sr_ucmerced', 'oli2msi', 'sr_so2sat']
    if args.dataset != "sent_nicfi":
        dataset_options.remove(args.dataset)

    auto_model = autoSR_RS(dataset_options,os.path.join(RESULTSDIR, "autosr","v1"),
                           f"{args.id}_autosr_v1_{args.dataset}", "1",max_trials=args.trials, scale=SCALE,overwrite=False)
    auto_model.fit(x=train_ds_arr["lr"], y=train_ds_arr["hr"], callbacks=[tensorboard_callback, tf.keras.callbacks.EarlyStopping(
        patience=10, min_delta=1e-4)], epochs=100, validation_data=(valid_ds_arr["lr"], valid_ds_arr["hr"]), verbose=2)
    results = np.asarray([auto_model.evaluate(
        test_ds_arr["lr"], test_ds_arr["hr"], custom_objects={"psnr": psnr, "ssim": ssim})])

    print(results)
    with open(RESULTSDIR+"/autosr/autosr.csv", "a") as f:
        f.write(
            f"{args.trials},v1,{args.dataset},{','.join([str(i) for i in results[0]])}\n")

    best_model = auto_model.export_model(
        custom_objects={"psnr": psnr, "ssim": ssim})
    print(best_model.summary())
    tf.saved_model.save(best_model,
        RESULTSDIR+f"/weights/autosr_v1_{args.dataset}_{args.trials}_{args.id}.h5")

    end = time.time()
    hours, rem = divmod(end-start_t, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
