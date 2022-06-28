from gc import callbacks
from typing import List, Dict
import autokeras as ak
import tensorflow as tf
from auto_models.blocks.autosr_block import AutoSR_Block
from utils import psnr, ssim


# TODO so you only need path to autokeras_weights, not the whole dir.
def autoSR_RS(dataset_options: List,
              results_dir: str, 
              project_name: str, 
              version: str, 
              max_trials: int = 1, 
              model_type=None,
              scale: int = 2,
              overwrite:bool=True,
              pretrained_weights_dir: Dict = {"wdsr":{
                                "cerrado":"/data1/s1620444/autokeras_weights/wdsr/cerrado",
                                "sr_so2sat": "/data1/s1620444/autokeras_weights/wdsr/sr_so2sat",
                                "sr_ucmerced": "/data1/s1620444/autokeras_weights/wdsr/sr_ucmerced",
                                "oli2msi": "/data1/s1620444/autokeras_weights/wdsr/oli2msi"
                            },
                      "rcan":{
                                "cerrado":"/data1/s1620444/autokeras_weights/rcan/cerrado",
                                "sr_so2sat": "/data1/s1620444/autokeras_weights/rcan/sr_so2sat",
                                "sr_ucmerced": "/data1/s1620444/autokeras_weights/rcan/sr_ucmerced",
                                "oli2msi": "/data1/s1620444/autokeras_weights/rcan/oli2msi"
                            }}):
    """ Constructs AutoSR-RS model
    inputs:
        results_dir: str        directory where results are saved
        project_name: str       project name used to make files & dirs
        # version: a | b | c | d  AutoSR-RS version to use
        max_trials: int         Number of trials for autokeras
        scale: int              Scaling factor of dataset used 
        overwrite: bool         whether to overwrite existing folder or not

    returns:
        ak.AutoModel
    """

    input_node = ak.ImageInput()
    output_node = AutoSR_Block(version, dataset_options,scale=scale, model_type=model_type,pretrained_weights_dir=pretrained_weights_dir)(input_node)
    output_node = ak.ImageHead(metrics=[psnr, ssim])(output_node)

    return ak.AutoModel(inputs=input_node,
                        outputs=output_node,
                        project_name=project_name+"_trial_"+str(max_trials),
                        overwrite=overwrite,
                        max_trials=max_trials,
                        directory=results_dir,
                        tuner="greedy",
                        distribution_strategy=tf.distribute.MirroredStrategy())
