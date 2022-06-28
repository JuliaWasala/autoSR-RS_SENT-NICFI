import os

import tensorflow as tf
import tensorflow_datasets as tfds

DATASETSDIR = os.environ["DATASETSDIR"]


def psnr(x, y, m_val=255):
    return tf.image.psnr(a=x, b=y, max_val=m_val)


def ssim(x, y, m_val=255):
    return tf.image.ssim(x, y, max_val=m_val)


def normalize(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def denormalize(x):
    return x * 255


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch


def evaluate(model, dataset, use_ssim=False):
    """ Evaluate PSNR of model on dataset 
    inputs:
        model:      keras model
        dataset:    tfds dataset
        use_ssim:   whether to calculate SSIM as well as PSNR
    """
    psnr_values = []
    ssim_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
        if use_ssim:
            ssim_val = ssim(hr, sr)[0]
            ssim_values.append(ssim_val)
    if use_ssim:
        return (tf.reduce_mean(psnr_values), tf.reduce_mean(ssim_values))
    return tf.reduce_mean(psnr_values)


def get_data(dataset_name: str, train_batch_size: int = -1, val_batch_size: int = -1, test_batch_size: int = -1, as_supervised: bool = False):
    """ Get tfds datasets for train,test,val
    inputs:
        dataset_name: cerrado |         name of dataset
                      sr_ucmerced | 
                      oli2msi | 
                      sr_so2sat | 
                      sent_nicfi
        train_batch_size: int=-1        batch size of train set
        val_batch_size: int=-1          batch size of val set
        test_batch_size: int=-1         batch size of test set
        as_supervised: bool=True        whether to load data as tuples (lr,hr)
                                        or dict {"hr": [...], "lr":[...]}

    returns:
        train, val, test: tfds datasets 
    """

    if dataset_name == "oli2msi":
        ds_size = 5325
        train_size = int(ds_size*0.64)
        test_size = int(ds_size*0.2)
        train_set = tfds.load(dataset_name,
                              split=f"train[:{train_size}]",
                              batch_size=train_batch_size,
                              as_supervised=as_supervised,
                              data_dir=DATASETSDIR)

        test_set = tfds.load(dataset_name,
                             split=f"train[{train_size}:{train_size+test_size}]",
                             batch_size=test_batch_size,
                             as_supervised=as_supervised,
                             data_dir=DATASETSDIR)

        valid_set = tfds.load(dataset_name,
                              split=f"train[{train_size+test_size}:]+test",
                              batch_size=val_batch_size,
                              as_supervised=as_supervised,
                              data_dir=DATASETSDIR)

    elif dataset_name == "sr_so2sat":
        ds_size = 376485
        train_size = int(ds_size*0.64)
        test_size = int(ds_size*0.2)

        train_set = tfds.load(dataset_name,
                              split=f"train[:{train_size}]",
                              batch_size=train_batch_size,
                              as_supervised=as_supervised,
                              data_dir=DATASETSDIR)

        test_set = tfds.load(dataset_name,
                             split=f"train[{train_size}:{train_size+test_size}]",
                             batch_size=test_batch_size,
                             as_supervised=as_supervised,
                             data_dir=DATASETSDIR)

        valid_set = tfds.load(dataset_name,
                              split=f"train[{train_size+test_size}:]+validation",
                              batch_size=val_batch_size,
                              as_supervised=as_supervised,
                              data_dir=DATASETSDIR)

    else:
        ds_sizes = {"cerrado": 1311, "sr_ucmerced": 2056,
                    "sr_bigearthnet": 590326,"sent_nicfi":12000}
        ds_size = ds_sizes[dataset_name]
        train_size = int(ds_size*0.64)
        test_size = int(ds_size*0.2)
        train_set = tfds.load(dataset_name,
                              split=f"train[:{train_size}]",
                              batch_size=train_batch_size,
                              as_supervised=as_supervised,
                              data_dir=DATASETSDIR)

        test_set = tfds.load(dataset_name,
                             split=f"train[{train_size}:{train_size+test_size}]",
                             batch_size=test_batch_size,
                             as_supervised=as_supervised,
                             data_dir=DATASETSDIR)

        valid_set = tfds.load(dataset_name,
                              split=f"train[{train_size+test_size}:]",
                              batch_size=val_batch_size,
                              as_supervised=as_supervised,
                              data_dir=DATASETSDIR)
    return test_set, valid_set, train_set
    
def get_test_set(dataset_name: str, test_batch_size: int = -1, as_supervised: bool = False):
    """ Get tfds dataset of the test split
    inputs:
        dataset_name: cerrado |         name of dataset
                      sr_ucmerced | 
                      oli2msi | 
                      sr_so2sat | 
                      sent_nicfi
        test_batch_size: int=-1         batch size of test set
        as_supervised: bool=True        whether to load data as tuples (lr,hr)
                                        or dict {"hr": [...], "lr":[...]}

    returns:
        test: tfds datasets 
    """

    if dataset_name == "oli2msi":
        ds_size = 5325
        train_size = int(ds_size*0.64)
        test_size = int(ds_size*0.2)
        test_set = tfds.load(dataset_name,
                             split=f"train[{train_size}:{train_size+test_size}]",
                             batch_size=test_batch_size,
                             as_supervised=as_supervised,
                             data_dir=DATASETSDIR)

    elif dataset_name == "sr_so2sat":
        ds_size = 376485
        train_size = int(ds_size*0.64)
        test_size = int(ds_size*0.2)

        test_set = tfds.load(dataset_name,
                             split=f"train[{train_size}:{train_size+test_size}]",
                             batch_size=test_batch_size,
                             as_supervised=as_supervised,
                             data_dir=DATASETSDIR)

    else:
        ds_sizes = {"cerrado": 1311, "sr_ucmerced": 2056,
                    "sr_bigearthnet": 590326,"sent_nicfi":12000}
        ds_size = ds_sizes[dataset_name]
        train_size = int(ds_size*0.64)
        test_size = int(ds_size*0.2)
        test_set = tfds.load(dataset_name,
                             split=f"train[{train_size}:{train_size+test_size}]",
                             batch_size=test_batch_size,
                             as_supervised=as_supervised,
                             data_dir=DATASETSDIR)

    return test_set