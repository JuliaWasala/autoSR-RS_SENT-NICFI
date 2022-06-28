# FROM https://github.com/hieubkset/Keras-Image-Super-Resolution

import sys
from typing import Optional, Union, Dict

import autokeras as ak
import tensorflow as tf
from autokeras.utils import utils
from keras_tuner.engine import hyperparameters
from tensorflow.keras.layers import ( Add, Conv2D, Input)
from tensorflow.keras.models import Model
from baselines.rcan.rcan import rir, upsample, generator

sys.setrecursionlimit(10000)

class RCANBlock(ak.Block):
    """ Block implementing RCAN with possible arguments to mutate
    inputs:
        pretrained_weights_dir: directory       directory with paths to pretrained weights, of shape {method: {dataset: path}}
        scale: int=2                            upscaling factor of datset
        n_rg:        int |                      number of residual groups
                     hyperparameters.Int
        n_res:      int |                       number of residual channel attention 
                     hyperparameters.Int        blocks per residual group
        resblock_type:  RB | RCAB | RG          residual block type
                     hyperparameters.Float
    """

    def __init__(self,
                 pretrained_weights_dir: Dict,
                 scale: Optional[int] = 2,
                 pretrained_on_dataset: Optional[Union[str, hyperparameters.Choice]] = None,
                 n_rg: Optional[Union[int, hyperparameters.Int]] = None,
                 n_res: Optional[Union[int, hyperparameters.Int]] = None,
        
                 **kwargs):

        super().__init__(**kwargs)

        self.scale = scale
        self.pretrained_weights_dir=pretrained_weights_dir

        self.n_rg = utils.get_hyperparameter(
            n_rg,
            hyperparameters.Int("n_rg", 2, 10, step=2, default=10),
            int,
        )
        self.n_res = utils.get_hyperparameter(
            n_res,
            hyperparameters.Int("n_res", 10, 20, step=2, default=20),
            int,
        )

        self.pretrained_on_dataset = utils.get_hyperparameter(
            pretrained_on_dataset,
            hyperparameters.Choice("pretrained_on_dataset", ["cerrado", "sr_ucmerced", "sr_so2sat", "oli2msi"]),
            str,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_rg": hyperparameters.serialize(self.n_rg),
                "n_res": hyperparameters.serialize(self.n_res),
                "scale":hyperparameters.serialize(self.scale),
                "pretrained_on_dataset": hyperparameters.serialize(self.pretrained_on_dataset),
                "pretrained_weights_dir":hyperparameters.serialize(self.pretrained_weights_dir),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["n_rg"] = hyperparameters.deserialize(config["n_rg"])
        config["n_res"] = hyperparameters.deserialize(config["n_res"])
        config["scale"] = hyperparameters.deserialize(config["scale"])
        config["pretrained_on_dataset"] = hyperparameters.deserialize(
            config["pretrained_on_dataset"])
        config["pretrained_weights_dir"]=hyperparameters.deserialize(config["pretrained_weights_dir"])
        return cls(**config)

    @staticmethod
    def get_weights(n_res,n_rg,dataset,pretrained_weights_dir):     
        """ Get pretrained weights from dataset with right shape (number of res blocks)"""

        pretrained_model_path=pretrained_weights_dir["rcan"][dataset]+"/model_weights.h5"
        if dataset =="oli2msi":
            scale=3
        else:
            scale=2
        model=generator(scale=scale)
        model.load_weights(pretrained_model_path)
        # get list
        complete_model_weights=[]
        for layer in model.layers:
            complete_model_weights.append(layer.get_weights())

        # remove normalize/denormalize layers: model_weights[1] en laatste
        model_weights=complete_model_weights[:1]+complete_model_weights[2:2022]

        n_extra_res=max(n_res-20,0)
        # add n_extra_res to each rg
        resblocks=[model_weights[i:i+min(200, 10*n_res)]+ \
                   model_weights[i+190:i+200]*n_extra_res+ \
                   model_weights[i+200:i+202] for i in range(2,min(2+202*n_rg,2022),202)]
        return model_weights[:2]+[r for group in resblocks for r in group]

    def build(self, hp, inputs=None):
        input_node = tf.nest.flatten(inputs)[0]

        n_rg = utils.add_to_hp(self.n_rg, hp)
        n_res = utils.add_to_hp(self.n_res, hp)
        dataset=utils.add_to_hp(self.pretrained_on_dataset,hp)
        weights=self.get_weights(n_res,n_rg,dataset,self.pretrained_weights_dir)

        print(f"n_rg = {n_rg}, n_res = {n_res}")

        model_input=Input(shape=(None,None,3))
        x = x_1 = Conv2D(64, 3,strides=1, padding='same')(model_input)
        x = rir(x, 64, n_rg=n_rg, n_rcab=n_res)
        x = Conv2D(64, 3,strides=1, padding='same')(x)
        x = Add()([x_1, x])
        x = upsample(x, 64, self.scale)
        x = Conv2D(64, 3,strides=1, padding='same')(x)

        output_node = Conv2D(64, 3,strides=1, padding='same')(x)
        model=Model(model_input, output_node)

        for i, layer_weights in enumerate(weights):
            model.layers[i].set_weights(layer_weights)
        return model(input_node)
