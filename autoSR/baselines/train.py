import os
import time
from functools import partial

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from utils import evaluate, get_data

from baselines.rcan.callbacks import make_lr_callback
from baselines.rcan.metrics import psnr
from baselines.rcan.rcan import generator as rcan
from baselines.rcan.utils import num_iter_per_epoch
from baselines.wdsr.wdsr import wdsr_b

DATADIR = os.environ["DATADIR"]
DATASETSDIR = os.environ["DATASETSDIR"]
RESULTSDIR = os.environ["RESULTSDIR"]


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def prepare_rcan_model(lr_init, n_rg=10,n_res=20,scale=2):
    """Compiles RCAN model
    inputs:
        lr_init: float      initial learning rate
        scale: int          upscaling factor of model
    returns:
        compiled keras model"""
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = rcan(scale=scale,n_rg=n_rg,n_res=n_res)
        loss = mean_absolute_error
        optimizer = Adam(learning_rate=lr_init)
        model.compile(optimizer=optimizer, loss=loss, metrics=[psnr])

    return model


def train_rcan(train, val, n_train, n_val, batch_size, exp_dir, n_rg=10,n_res=20,scale=2, init_epoch=0, epochs=20, lr_init=1e-4, lr_decay=0.5, lr_decay_at_steps=[10, 15]):
    """Trains RCAN model
    inputs:
        train: tfds         training data: batch_size=batch_size
                                           repeat: None
                                           as_supervised: True (in tuples (lr,hr))
        val: tfds           validation data: same as training but with batch_size =1
        n_train: int        number of training examples
        n_val: int          number of validation examples
        batch_size: int     training batch size
        exp dir: str        path to dir where to save weights
        n_rg: int=10        number of residual groups
        n_res: int=20       number of residual blocks per group
        scale: int=2        model upscaling factor
        init_epoch: int=0   epoch at which to start training
        epochs: int=20      number of epochs to train
        lr_init: int=1e-4   initial learning rate
        lr_decay: int=0.5   learning rate decay
        lr_decay_at_steps:  when learning rate should decay
            list=[10,15]
    returns:
        trained keras model"""

    print("** Loading training images")
    start = time.time()
    print("Finish loading images in %.2fs" % (time.time() - start))

    gpu_model = prepare_rcan_model(lr_init, n_rg=n_rg,n_res=n_res,scale=scale)
    lr_callback = make_lr_callback(lr_init, lr_decay, lr_decay_at_steps)

    gpu_model.fit(train, epochs=epochs,
                  steps_per_epoch=num_iter_per_epoch(n_train, batch_size),
                  callbacks=[lr_callback, EarlyStopping(
                      monitor='val_loss', patience=10)],
                  initial_epoch=init_epoch,
                  validation_data=val,
                  validation_steps=n_val)

    gpu_model.save_weights(os.path.join(exp_dir, 'final_model.h5'))

    return gpu_model


def prepare_wdsr_model(scale=2):
    """Compiles WDSR model
    inputs:
        scale: int          upscaling factor of model
    returns:
        compiled keras model"""
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = wdsr_b(scale=scale, num_res_blocks=32)
        optimizer = Adam(learning_rate=PiecewiseConstantDecay(
            boundaries=[200000], values=[1e-3, 5e-4]))
        model.compile(optimizer=optimizer,
                      loss=mean_absolute_error, metrics=[psnr])

    return model


def train_wdsr(train, val, n_train, n_val, batch_size, exp_dir, scale=2, init_epoch=0, epochs=20):
    """Trains WDSR model
    inputs:
        train: tfds         training data: batch_size=batch_size
                                           repeat: None
                                           as_supervised: True (in tuples (lr,hr))
        val: tfds           validation data: same as training but with batch_size =1
        n_train: int        number of training examples
        n_val: int          number of validation examples
        batch_size: int     training batch size
        exp dir: str        path to dir where to save weights
        scale: int=2        model upscaling factor
        init_epoch: int=0   epoch at which to start training
        epochs: int=20      number of epochs to train
        lr_init: int=1e-4   initial learning rate
    returns:
        trained keras model"""
    print("** Loading training images")
    start = time.time()
    print("Finish loading images in %.2fs" % (time.time() - start))

    gpu_model = prepare_wdsr_model(scale=scale)

    gpu_model.fit(train, epochs=epochs,
                  steps_per_epoch=num_iter_per_epoch(n_train, batch_size),
                  callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
                  initial_epoch=init_epoch,
                  validation_data=val,
                  validation_steps=n_val)

    gpu_model.save_weights(os.path.join(exp_dir, 'final_model.h5'))

    return gpu_model


def train_test(baseline_name: str, dataset_name: str, epochs: int, id: str) -> None:
    """ Train & eval baseline. Write results to file.
    inputs:
        baseline_name: wdsr |           baseline algorithm
                       rcan |
                       rcan_undeep 
        dataset_name: cerrado |         dataset to train on
                      sr_ucmerced |
                      oli2msi |
                      sr_so2sat|
                      sent_nicfi
        epochs: int                     number of epochs to train
        id: int                         run id for logging


    """

    if dataset_name == "oli2msi":
        scale = 3
        if baseline_name == "rcan":
            train_batch_size = 4
        elif baseline_name == "wdsr" or baseline_name=="rcan_undeep":
            train_batch_size = 16
    elif dataset_name == "sr_ucmerced" and baseline_name == "rcan":
        train_batch_size = 4
        scale = 2
    elif baseline_name =="rcan" and dataset_name=="sent_nicfi":
        train_batch_size=8
        scale=2
    else:
        scale = 2
        train_batch_size = 16

    print(dataset_name)
    print(baseline_name)
    print(train_batch_size)

    if baseline_name == "rcan":
        train_func = train_rcan
    elif baseline_name == "wdsr":
        train_func = train_wdsr
    elif baseline_name == "rcan_undeep":
        train_func = partial(train_rcan, n_rg=1,  n_res=32)

    weights_dir = f'{RESULTSDIR}/weights/{baseline_name}-{dataset_name}-x{scale}-{id}'
    os.makedirs(weights_dir, exist_ok=True)

    print("Load data ...")
    test_ds, valid_ds, train_ds = get_data(dataset_name,
                                           train_batch_size=train_batch_size,
                                           test_batch_size=1,
                                           val_batch_size=1,
                                           as_supervised=True)
    num_train = tf.get_static_value(train_ds.__len__())*train_batch_size
    num_valid = tf.get_static_value(valid_ds.__len__())
    train_ds = train_ds.repeat(None)
    valid_ds = valid_ds.repeat(None)
    test_ds = test_ds.repeat(1)

    print(f"Train {baseline_name}...")
    model = train_func(train_ds,
                       valid_ds,
                       num_train,
                       num_valid,
                       train_batch_size,
                       weights_dir,
                       epochs=epochs,
                       scale=scale)

    # load model & eval
    psnr, ssim = evaluate(model, test_ds, use_ssim=True)
    print(f'PSNR={psnr.numpy():3f}')
    print(f'SSIM={ssim.numpy():3f}')

    # write results.
    results_file = f'{RESULTSDIR}/baselines/{baseline_name}/{baseline_name}-{dataset_name}-x{scale}-{id}_results.csv'
    with open(results_file, "w") as f:
        f.write(f'{psnr.numpy()}, {ssim.numpy()}')
