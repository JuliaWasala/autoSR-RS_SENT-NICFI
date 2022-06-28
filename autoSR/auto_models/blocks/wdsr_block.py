
from typing import Optional, Dict, Union

import autokeras as ak
import tensorflow as tf
from autokeras.utils import utils
from baselines.wdsr.utils import pixel_shuffle
from baselines.wdsr.wdsr import conv2d_weightnorm
from baselines.wdsr.wdsr import res_block_b,wdsr_b
from keras_tuner.engine import hyperparameters
from tensorflow.python.keras.layers import Add, Lambda,Input
from tensorflow.python.keras.models import Model


class WDSRBlock(ak.Block):
    """ Block implementing WDSR with possible arguments to mutate
    inputs:
        pretrained_weights_dir: dictionary      dictionary containing paths to pretrained weights, of shape {method:{dataset:path}}
        scale: int=2                            upscaling factor of dataset
        n_res:      int |                       number of residual blocks 
                     hyperparameters.Int        
        resblock_type:   RB | RCAB | RG         residual block type to use      
                     hyperparameters.Float
    """

    def __init__(self,
                 pretrained_weights_dir: Dict,
                 scale: Optional[int] = 2,
                 pretrained_on_dataset: Optional[Union[str, hyperparameters.Choice]]=None,
                 n_res: Optional[Union[int, hyperparameters.Int]] = None,
                 **kwargs):

        super().__init__(**kwargs)
        self.pretrained_weights_dir=pretrained_weights_dir
        self.scale = scale

        self.n_res = utils.get_hyperparameter(
            n_res,
            hyperparameters.Int("n_res", 4, 40, step=2, default=32),
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
                "n_res": hyperparameters.serialize(self.n_res),
                "pretrained_on_dataset":hyperparameters.serialize(self.pretrained_on_dataset),
                "scale":hyperparameters.serialize(self.scale),
                "pretrained_weights_dir":hyperparameters.serialize(self.pretrained_weights_dir),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["n_res"] = hyperparameters.deserialize(config["n_res"])
        config["pretrained_on_dataset"] = hyperparameters.deserialize(config["pretrained_on_dataset"])
        config["scale"]=hyperparameters.deserialize(config["scale"])
        config["pretrained_weights_dir"]=hyperparameters.deserialize(config["pretrained_weights_dir"])
        return cls(**config)
    
    @staticmethod
    def get_weights(n_res,dataset,pretrained_weights_dir):
        """ Get pretrained weights from dataset with right shape (number of res blocks)
        It returns the weights up until the last residual block, not including the rest."""
        pretrained_model_path=pretrained_weights_dir["wdsr"][dataset]+"/model_weights.h5"
        if dataset=="oli2msi":
            scale=3
        else:
            scale=2
        model=wdsr_b(scale=scale,num_res_blocks=32)
        model.load_weights(pretrained_model_path)
        # get list

        complete_model_weights=[]
        for layer in model.layers:
            complete_model_weights.append(layer.get_weights())

        # remove normalize/denormalize layers: model_weights[1] en laatste
        model_weights=complete_model_weights[:1]+complete_model_weights[2:130]
        if n_res>=32:
            return model_weights
        elif n_res <32:
            # remove surplus res blocks
            return model_weights[:2]+model_weights[2: 2+4*n_res]


    def build(self, hp, inputs=None):
        input_node = tf.nest.flatten(inputs)[0]

        n_res = utils.add_to_hp(self.n_res, hp)
        dataset=utils.add_to_hp(self.pretrained_on_dataset,hp)
        weights=self.get_weights(n_res, dataset,self.pretrained_weights_dir)

        model_input = Input(shape=(None, None, 3))
        m = conv2d_weightnorm(32, 3, padding='same')(model_input)

        for _ in range(n_res):
            m = res_block_b(m, 32,6,3,scaling=None)

        m = conv2d_weightnorm(3 * self.scale ** 2, 3, padding='same',
                 name=f'conv2d_main_scale_{self.scale}')(m)
        m = Lambda(pixel_shuffle(self.scale))(m)

        # skip branch
        s = conv2d_weightnorm(3 * self.scale ** 2, 5, padding='same',
                 name=f'conv2d_skip_scale_{self.scale}')(model_input)
        s = Lambda(pixel_shuffle(self.scale))(s)

        output_node = Add()([m, s])
        model=Model(model_input, output_node)

        for i, layer_weights in enumerate(weights):
            model.layers[i].set_weights(layer_weights)
        return model(input_node)
