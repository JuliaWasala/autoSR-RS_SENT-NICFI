import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tqdm import tqdm,trange

import tensorflow_datasets as tfds
import sys
import numpy as np

WEIGHTSDIR="/home/s1620444/data1/results/weights"
PROJECTDIR="/data1/s1620444/automl-sr-rs/autoSR"

DATASETSDIR=os.environ["DATASETSDIR"]
RESULTSDIR=os.environ["RESULTSDIR"]
DATADIR=os.environ["DATADIR"]

sys.path.append(PROJECTDIR)
from utils import get_data,resolve
from baselines.wdsr.wdsr import wdsr_b
from utils import psnr,ssim
from baselines.rcan.rcan import generator as rcan
from auto_models.models.autosrcnn import autoSRCNN
from auto_models.models.autosr_rs import autoSR_RS

# load wdsr & rcan
def load_model(model_name: str, dataset: str, id: int, weights_dir: str):
    if dataset == "oli2msi":
        SCALE=3
    else:
        SCALE=2

    dataset_options=["cerrado","sr_ucmerced","sr_so2sat","oli2msi"]

    if model_name=="wdsr":
        model=wdsr_b(num_res_blocks=32, scale=SCALE)
        model.load_weights(f"{weights_dir}/wdsr-b-{dataset}-32-x{SCALE}-{id}/final_model.h5")
    elif model_name=="rcan":
        model=rcan(scale=SCALE)
        # extract weights from nested checkpoint structure
        model_dir=f"{weights_dir}/rcan-{dataset}-x{SCALE}-{id}"
        checkpoint_subdirs=os.listdir(model_dir)
        checkpoint_subdirs.sort()
        # sometimes there are no checkpoints, just the final model
        if checkpoint_subdirs[-1] == "final_model.h5":
            subdir=""
        else:
            subdir=checkpoint_subdirs[-1]
        model.load_weights(os.path.join(model_dir,subdir, "final_model.h5"))
    elif model_name=="autosrcnn":
        model=autoSRCNN(os.path.join(RESULTSDIR,"autosrcnn"), f"{id}_simple_srcnn_{dataset}", max_trials=20, scale=SCALE, overwrite=False)
    elif model_name=="autosr":
        if dataset in dataset_options:
            dataset_options.remove(dataset)
            # autoSR_RS(dataset_options,trained_weights_dir,f"0_autosr_v1_{dataset}", "v1",max_trials=20, scale=scale, overwrite=False)
        model=autoSR_RS(dataset_options,os.path.join(RESULTSDIR,"autosr/v1"), f"{id}_autosr_v1_{dataset}","v1", max_trials=20, scale=SCALE, overwrite=False)
    return model

def evaluate_model(model, model_name, dataset):
    if model_name =="autosrcnn" or model_name=="autosr":
        predictions=[]
        predictions.append(model.predict(dataset["lr"], custom_objects={"psnr":psnr,"ssim":ssim}))
        clipped_predictions=[np.clip(img,0,255) for img in predictions]
        rounded_predictions = [np.around(img) for img in clipped_predictions]
        int_predictions= [img.astype("uint8") for img in rounded_predictions]

        # eval
        tf_predictions=tf.stack([tf.convert_to_tensor(im, np.uint8) for im in int_predictions])
        gt=tf.stack([tf.convert_to_tensor(im, np.uint8) for im in dataset["hr"]])

        return psnr(tf_predictions,gt).numpy(),ssim(tf_predictions,gt).numpy()
        
    # rcan & wdsr
    predictions=[]
    psnr_values=[]
    ssim_values=[]
    for lr, hr in dataset:
        sr= resolve(model, lr)
        psnr_values.append(tf.keras.backend.get_value(psnr(hr,sr))[0])
        ssim_values.append(tf.keras.backend.get_value(ssim(hr,sr))[0])

    return np.array(psnr_values), np.array(ssim_values)


def get_test_set(dataset_name,batch_size, as_supervised, as_numpy):
    test,_,_=get_data(dataset_name,train_batch_size=-1, test_batch_size=batch_size, val_batch_size=-1, as_supervised=as_supervised)
    if as_numpy:
        return tfds.as_numpy(test)
    else:
        test.repeat(1)
        return test

if __name__=="__main__":
    datasets=[ "cerrado","sr_ucmerced","oli2msi", "sent_nicfi"]
    models=["autosr","autosrcnn"]

    datasets_bar=tqdm(datasets)
    datasets_bar.set_description("train_data")
    models_bar=tqdm(models)
    models_bar.set_description("model")

    with open(f"{RESULTSDIR}/scores_per_img_auto.csv", "w") as f:
        f.write("model,dataset,psnr,ssim\n")
        for model_name in models_bar:
            if model_name=="autosrcnn" or model_name=="autosr":
                batch_size=-1
                as_supervised=False
                as_numpy=True
            else:
                batch_size=1
                as_supervised=True
                as_numpy=False
            for dataset in datasets_bar:
                test = get_test_set(dataset,batch_size, as_supervised,as_numpy)
                model=load_model(model_name, dataset, 0, WEIGHTSDIR)
                psnr_vals,ssim_vals= evaluate_model(model, model_name, test)
                print(model_name,dataset, psnr_vals, ssim_vals)
                f.write(",".join([model_name, dataset, "["+",".join(str(i) for i in psnr_vals)+"]","["+",".join(str(i) for i in ssim_vals)+"]"])+"\n")

