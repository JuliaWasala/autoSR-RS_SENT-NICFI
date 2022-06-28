from typing import Optional, Union

import autokeras as ak
import tensorflow as tf
import tensorflow_addons as tfa
from autokeras.utils import utils
from baselines.wdsr.utils import pixel_shuffle
from keras_tuner.engine import hyperparameters
from tensorflow.python.keras.layers import Add, Conv2D, Lambda

CONV_TYPE = "conv_type"
CONV2D = "conv2d"
CONV2D_WEIGHTNORM = "conv2d_weightnorm"


def res_block_b(x_in, num_filters, expansion, kernel_size, scaling, conv):
    """ Residual block
    inputs:
        x_in: tensor
        num_filters: int
        expansion: int      residual block expansion
        kenrel_size: int
        scaling:            residual block scaling
        conv: Conv2D | Conv2D_weightnorm

    returns:
        tensor
    """
    linear = 0.8
    x = conv(num_filters * expansion, 1,
             padding='same', activation='relu')(x_in)
    x = conv(int(num_filters * linear), 1, padding='same')(x)
    x = conv(num_filters, kernel_size, padding='same')(x)
    if scaling > 0:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)


class WDSRBlock(ak.Block):
    """ Block implementing WDSR with possible arguments to mutate
    inputs:
        scale: int=2                            upscaling factor of datset
        kernel_size: int |                      kernel size
                     hyperparameters.Choice
        filters:     int |                      number of filters
                     hyperparameters.Choice
        n_res:      int |                       number of residual blocks 
                     hyperparameters.Int        
        r_scaling:   float |                    residual block scaling
                     hyperparameters.Float
        expansion:   int |                      res block expansion
                     hyperparameters.Int
        diversify_res: bool = False             whether each residual block has own set
                                                parameters (True) or shared parameters (False)
        convs_optional: bool = False            whether first & last conv are optional AND
                                                whether Conv2D can be swapped for
                                                weight normalized Conv2D
    """

    def __init__(self,
                 scale: Optional[int] = 2,
                 kernel_size: Optional[Union[int, hyperparameters.Choice]] = None,
                 filters: Optional[Union[int, hyperparameters.Choice]] = None,
                 n_res: Optional[Union[int, hyperparameters.Int]] = None,
                 expansion: Optional[Union[int, hyperparameters.Int]] = None,
                 r_scaling: Optional[bool] = None,
                 diversify_res: Optional[bool] = False,
                 convs_optional: Optional[bool] = False,
                 **kwargs):

        super().__init__(**kwargs)

        self.scale = scale

        self.kernel_size = utils.get_hyperparameter(
            kernel_size,
            hyperparameters.Choice("kernel_size", [3, 5, 7, 9], default=3),
            int,
        )
        self.filters = utils.get_hyperparameter(
            filters,
            hyperparameters.Choice(
                "filters", [16, 32, 64, 128, 256, 512], default=64),
            int,
        )
        self.n_res = utils.get_hyperparameter(
            n_res,
            hyperparameters.Int("n_res", 4, 40, step=2, default=32),
            int,
        )
        self.expansion = utils.get_hyperparameter(
            expansion,
            hyperparameters.Int("expansion", 4, 32, step=2, default=16),
            int,
        )
        self.r_scaling = utils.get_hyperparameter(
            r_scaling,
            hyperparameters.Float("r_scaling", 0.1, 1,
                                  step=0.1, default=1),
            int,
        )
        self.diversify_res = diversify_res
        self.convs_optional = convs_optional

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": hyperparameters.serialize(self.kernel_size),
                "filters": hyperparameters.serialize(self.filters),
                "n_res": hyperparameters.serialize(self.n_res),
                "expansion": hyperparameters.serialize(self.expansion),
                "r_scaling": hyperparameters.serialize(self.r_scaling),
                "scale": self.scale,
                "diversify_res": self.diversify_res,
                "convs_optional": self.convs_optional,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["kernel_size"] = hyperparameters.deserialize(
            config["kernel_size"])
        config["filters"] = hyperparameters.deserialize(config["filters"])
        config["n_res"] = hyperparameters.deserialize(config["n_res"])
        config["expansion"] = hyperparameters.deserialize(config["n_rcab"])
        config["r_scaling"] = hyperparameters.deserialize(config["r_scaling"])
        config["scale"] = hyperparameters.deserialize(config["scale"])
        config["diversify_res"] = hyperparameters.deserialize(
            config["diversify_res"])
        config["convs_optional"] = hyperparameters.deserialize(
            config["convs_optional"])
        return cls(**config)

    @staticmethod
    def _get_conv(conv_type):
        if conv_type == CONV2D:
            return Conv2D
        elif conv_type == CONV2D_WEIGHTNORM:
            return conv2d_weightnorm

    def build(self, hp, inputs=None):
        input_node = tf.nest.flatten(inputs)[0]

        kernel_size = utils.add_to_hp(self.kernel_size, hp)
        filters = utils.add_to_hp(self.filters, hp)
        n_res = utils.add_to_hp(self.n_res, hp)

        if self.convs_optional:
            conv_type = hp.Choice(
                CONV_TYPE, [CONV2D, CONV2D_WEIGHTNORM], default=CONV2D_WEIGHTNORM)
            final_conv = hp.Boolean('final_conv')
        else:
            conv_type = CONV2D_WEIGHTNORM
            final_conv = False

        conv = self._get_conv(conv_type)

        m = conv(filters, kernel_size, padding='same')(input_node)

        if not self.diversify_res:
            expansion = utils.add_to_hp(self.expansion, hp)
            r_scaling = utils.add_to_hp(self.r_scaling, hp)

            for _ in range(n_res):
                m = res_block_b(m, filters, expansion,
                                kernel_size, r_scaling, conv)

        else:
            for i in range(n_res):
                m = res_block_b(m, filters, utils.add_to_hp(self.expansion, hp, "expansion_{i}".format(
                    i=i)), utils.add_to_hp(self.kernel_size, hp, "kernel_size_{i}".format(i=i)), utils.add_to_hp(self.r_scaling, hp, "r_scaling_{i}".format(i=i)), conv)

        m = conv(3 * self.scale ** 2, 3, padding='same',
                 name=f'conv2d_main_scale_{self.scale}')(m)
        m = Lambda(pixel_shuffle(self.scale))(m)

        # skip branch
        s = conv(3 * self.scale ** 2, 5, padding='same',
                 name=f'conv2d_skip_scale_{self.scale}')(input_node)
        s = Lambda(pixel_shuffle(self.scale))(s)

        output_node = Add()([m, s])

        # optional conv at end
        if final_conv:
            output_node = conv(filters, kernel_size,
                               padding='same')(output_node)

        return output_node
