import autokeras as ak
import keras_tuner as kt
import tensorflow as tf
from auto_models.blocks.blocks import (Denormalize01Block, Normalize01Block,
                                       UpscalingBlock)
from utils import psnr, ssim


def autoSRCNN(results_dir: str, project_name: str, max_trials: str = 1, scale: str = 2, overwrite=True):
    """ Constructs an AutoModel inspired by SRCNN
    inputs:
        results_dir: str        dir to save results
        project_name: str       name used to create files & dirs
        max_trials: int = 1     max number of trials
        scale: int = 2:         upscaling factor
        overwrite: bool = True  overwrite source dir

    returns:
        ak.AutoModel
    """
    filters = kt.engine.hyperparameters.Choice(
        "filters", [3, 64, 128],
    )
    kernels = kt.engine.hyperparameters.Choice(
        "kernels", [1, 3, 5, 9],
    )

    input_node = ak.ImageInput()
    output_node = Normalize01Block()(input_node)
    output_node = UpscalingBlock(
        scale=scale, upscale_type="interpolation")(output_node)
    output_node = ak.ConvBlock(
        max_pooling=False, dropout=0, filters=filters, kernel_size=kernels, same_padding=True)(output_node)
    output_node = Denormalize01Block()(output_node)
    output_node = ak.ImageHead(metrics=[psnr, ssim])(output_node)

    return ak.AutoModel(inputs=input_node,
                        outputs=output_node,
                        project_name=project_name+"_trial_"+str(max_trials),
                        overwrite=overwrite,
                        max_trials=max_trials,
                        directory=results_dir,
                        tuner="greedy",
                        distribution_strategy=tf.distribute.MirroredStrategy())
