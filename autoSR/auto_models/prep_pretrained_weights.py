
import os
import pickle
import sys

from tensorflow import keras

sys.path.append("/data1/s1620444/automl-sr-rs/autoSR")
from baselines.rcan.rcan import generator
from baselines.wdsr.wdsr import wdsr_b


def save_pickle(filename,object):
    """ saves file as pickle
    inputs:
        object: any object
        filename: str   destination path to save pickle
    """
    with open(filename, "wb") as f:
        pickle.dump(object,f)

def save_wdsr_modules(model, weights_folder):
    """ Saves the weights of wdsr modules as h5 or pkl:
        - complete model (h5)
        - skip_upscale: upscale module of skip branch (pkl)
        - main_conv: convolution in main branch (pkl)
        - resblocks: residual blocks in main branch (pkl)
        - main_upscale: upscale module of main branch (pkl)
    inputs:
        model: instantiated wdsr_b model of depth 32
        weights_folder: str         path where to save weights
    """
    skip_upscale=[model.layers[132].get_weights(),model.layers[134].get_weights()]
    main_conv=[model.layers[2].get_weights()]
    resblocks = [[l.get_weights() for l in model.layers[i:i+4]] for i in range(3,131,4)]
    main_upscale=[model.layers[131].get_weights(),model.layers[133].get_weights()]

    save_pickle(f"{weights_folder}/skip_upscale.pkl", skip_upscale)
    save_pickle(f"{weights_folder}/main_conv.pkl", main_conv)
    save_pickle(f"{weights_folder}/resblocks.pkl", resblocks)
    save_pickle(f"{weights_folder}/main_upscale.pkl", main_upscale)

def save_rcan_modules(model, weights_folder):
    """ Saves the weights of rcan modules as h5 or pkl:
        - complete model (h5)
        - conv1: first convolution (pkl)
        - rir: residual-in-residual structure (pkl)
        - rg: weights of the residual groups (pkl)
        - rcab: weights of all rcab modules (pkl)
        - conv2: second convolution (pkl)
        - upscale: upscale module (pkl)
        - conv3: third and final convolution (pkl)
    inputs:
        model: instantiated rcan model with n_rg=10, n_rcab=20
        weights_folder: str         path where to save weights
    """
    conv1=[model.layers[2].get_weights()]
    rir=[l.get_weights() for l in model.layers[3:2025]]
    rg=[ rir[i:i+202] for i in range(0,2020,202)]
    rcab = [[r[i:i+10] for i in range(0,200,10)] for r in rg]
    conv2 = [model.layers[2025].get_weights()]
    upscale = [l.get_weights() for l in model.layers[2027:2039]]
    conv3=[model.layers[2030].get_weights()]

    save_pickle(f"{weights_folder}/conv1.pkl", conv1)
    save_pickle(f"{weights_folder}/rir.pkl", rir)
    save_pickle(f"{weights_folder}/rg.pkl", rg)
    save_pickle(f"{weights_folder}/rcab.pkl", rcab)
    save_pickle(f"{weights_folder}/conv2.pkl", conv2)
    save_pickle(f"{weights_folder}/upscale.pkl", upscale)
    save_pickle(f"{weights_folder}/conv3.pkl", conv3)

def save_model_weights(model_name,pretrained_weights_paths,dest_dir):
    """Saves the pretrained weights of a model of different datasets. 
    See save_wdsr_modules and save_rcan_modules for details
    inputs:
        model_name: wdsr | rcan
        pretrained_weights: dict    structure: {rcan: {dataset: path,...}, wdsr: {dataset: path,...}}
                                    paths to pretrained models
                                    where dataset: cerrado | sr_ucmerced | sr_so2sat | oli2msi
        dest_dir:  str              path where to save weights of modules
    """
    for dataset in pretrained_weights_paths[model_name]:
        if dataset=="oli2msi":
            scale=3
        else:
            scale=2

        if model_name=="wdsr":
            model = wdsr_b(scale=scale,num_res_blocks=32)
        elif model_name=="rcan":
            model=generator(scale=scale)
        else:
            raise ValueError(f"Invalid model_name {model_name}")

        weights_path=pretrained_weights_paths[model_name][dataset]
        model.load_weights(weights_path)
        weights_folder=f"{dest_dir}/{model_name}/{dataset}"
        model.save_weights(f"{weights_folder}/model_weights.h5")

        if model_name =="wdsr":
            save_wdsr_modules(model,weights_folder)

        elif model_name=="rcan":
            print("save rcan_modules")
            save_rcan_modules(model,weights_folder)

if __name__=="__main__":

    WEIGHTSDIR=os.environ["RESULTSDIR"]+"/weights"
    pretrained_models = {"rcan":{"cerrado":f"{WEIGHTSDIR}/rcan-cerrado-x2-0/rcan-03-20-00:11/final_model.h5",
                            "sr_ucmerced":f"{WEIGHTSDIR}/rcan-sr_ucmerced-x2-0/rcan-03-22-19:03/final_model.h5",
                            "sr_so2sat":f"{WEIGHTSDIR}/rcan-sr_so2sat-x2-0/rcan-03-31-14:49/final_model.h5",
                            "oli2msi":f"{WEIGHTSDIR}/rcan-oli2msi-x3-0/rcan-03-22-11:35/final_model.h5",
                            },
                    "wdsr":{"cerrado": f"{WEIGHTSDIR}/wdsr-b-cerrado-32-x2-0/final_model.h5",
                            "sr_ucmerced": f"{WEIGHTSDIR}/wdsr-b-sr_ucmerced-32-x2-0/final_model.h5",
                            "sr_so2sat":f"{WEIGHTSDIR}/wdsr-b-sr_so2sat-32-x2-0/final_model.h5",
                            "oli2msi":f"{WEIGHTSDIR}/wdsr-b-oli2msi-32-x3-0/final_model.h5",
    }}
    AUTOKERAS_WEIGHTS_PATH="/data1/s1620444/autokeras_weights"
    for model_name in ["rcan", "wdsr"]:
        save_model_weights(model_name, pretrained_models, AUTOKERAS_WEIGHTS_PATH)
