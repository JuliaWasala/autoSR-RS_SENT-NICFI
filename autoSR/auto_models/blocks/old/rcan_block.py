# FROM https://github.com/hieubkset/Keras-Image-Super-Resolution

import sys
from typing import Optional, Union

import autokeras as ak
import tensorflow as tf
import tensorflow_addons as tfa
from autokeras.utils import utils
from keras_tuner.engine import hyperparameters
from tensorflow.keras.layers import (Activation, Add, Conv2D, Dense,
                                     GlobalAveragePooling2D, Lambda, Multiply,
                                     Reshape)

sys.setrecursionlimit(10000)

CONV_TYPE = "conv_type"
CONV2D = "conv2d"
CONV2D_WEIGHTNORM = "conv2d_weightnorm"


def sub_pixel_conv2d(scale, **kwargs):
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)


def upsample(input_tensor, filters, scale):
    """ Upsampling module
    inputs: 
        input_tensor: tensor
        filters: int
        scale: int

    returns:
        tensor
    """
    x = Conv2D(filters=filters * scale * scale, kernel_size=3,
               strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv2d(scale)(x)
    x = Activation('relu')(x)
    return x


def ca(input_tensor, filters, reduce=16):
    """ Channel attention module
    inputs:
        input_tensor: tensor
        filters: int
        reduce: int = 16

    returns:
        tensor
    """
    x = GlobalAveragePooling2D()(input_tensor)
    x = Reshape((1, 1, filters))(x)
    x = Dense(filters/reduce,  activation='relu',
              kernel_initializer='he_normal', use_bias=False)(x)
    x = Dense(filters, activation='sigmoid',
              kernel_initializer='he_normal', use_bias=False)(x)
    x = Multiply()([x, input_tensor])
    return x


def rcab(input_tensor, filters, kernel_size, conv, scale=0.1):
    x = conv(filters, kernel_size,
             strides=1, padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = conv(filters, kernel_size, strides=1, padding='same')(x)
    x = ca(x, filters)
    if scale > 0:
        x = Lambda(lambda t: t * scale)(x)
    x = Add()([x, input_tensor])

    return x


def rg(input_tensor, filters, kernel_size, conv, n_rcab=20, scale=0.1):
    """ Residual group block
    inputs:
        input_tensor: tensor
        filters: int
        kernel_size: int
        conv: Conv2D | Conv2D_weightnorm
        n_rcab: int             number of residual channel attention blocks
        scale: float = 0.1      residual block scaling
    """
    x = input_tensor
    for _ in range(n_rcab):
        x = rcab(x, filters, kernel_size, conv, scale=scale)
    x = conv(filters, kernel_size, strides=1, padding='same')(x)
    x = Add()([x, input_tensor])

    return x


def rir(input_tensor, filters, kernel_size, conv, n_rg=10, n_rcab=20, scale=0.1):
    """ Residual-in-residual block
    inputs:
        input_tensor: tensor
        filters: int
        kernel_size: int
        conv: Conv2D | Conv2D_weightnorm
        n_rg: int = 10      number of residual groups
        n_rcab: int=20      number of residual channel attention blocks
        scale: float = 0.1  residual block scaling

    returns:
        tensor
    """

    x = input_tensor
    for i in range(n_rg):
        x = rg(x, filters, kernel_size, conv, n_rcab=n_rcab, scale=scale)
    x = conv(filters=filters, kernel_size=kernel_size,
             strides=1, padding='same')(x)
    x = Add()([x, input_tensor])

    return x


def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None, **kwargs):
    return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, activation=activation, **kwargs), data_init=False)


class RCANBlock(ak.Block):
    """ Block implementing RCAN with possible arguments to mutate
    inputs:
        scale: int=2                            upscaling factor of datset
        kernel_size: int |                      kernel size
                     hyperparameters.Choice
        filters:     int |                      number of filters
                     hyperparameters.Choice
        n_rg:        int |                      number of residual groups
                     hyperparameters.Int
        n_rcab:      int |                      number of residual channel attention 
                     hyperparameters.Int        blocks per residual group
        r_scaling:   float |                    residual channel attention block scaling
                     hyperparameters.Float
        diversify_res: bool = False             whether each res group & channel attention
                                                block has own set of parameters (True)
                                                or shared parameters (False) 
        convs_optional: bool = False            whether first & last conv are optional AND
                                                whether Conv2D can be swapped for
                                                weight normalized Conv2D
    """

    def __init__(self,
                 scale: Optional[int] = 2,
                 kernel_size: Optional[Union[int, hyperparameters.Choice]] = None,
                 filters: Optional[Union[int, hyperparameters.Choice]] = None,
                 n_rg: Optional[Union[int, hyperparameters.Int]] = None,
                 n_rcab: Optional[Union[int, hyperparameters.Int]] = None,
                 r_scaling: Optional[Union[float, hyperparameters.Float]] = None,
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
        self.n_rg = utils.get_hyperparameter(
            n_rg,
            hyperparameters.Int("n_rg", 2, 16, step=2, default=10),
            int,
        )
        self.n_rcab = utils.get_hyperparameter(
            n_rcab,
            hyperparameters.Int("n_rcab", 10, 40, step=2, default=20),
            int,
        )
        self.r_scaling = utils.get_hyperparameter(
            r_scaling,
            hyperparameters.Float("r_scaling", 0, 1, step=0.1, default=0.1),
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
                "n_rg": hyperparameters.serialize(self.n_rg),
                "n_rcab": hyperparameters.serialize(self.n_rcab),
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
        config["n_rg"] = hyperparameters.deserialize(config["n_rg"])
        config["n_rcab"] = hyperparameters.deserialize(config["n_rcab"])
        config["r_scaling"] = hyperparameters.deserialize(config["r_scaling"])
        config["scale"] = hyperparameters.deserialize(config["scale"])
        config["diversify_res"] = hyperparameters.deserialize(
            config["diversify_res"])
        config["convs_optional"] = hyperparameters.deserialize(
            config["convs_optional"])
        return cls(**config)

    def _get_conv(conv_type):
        if conv_type == CONV2D:
            return Conv2D
        elif conv_type == CONV2D_WEIGHTNORM:
            return conv2d_weightnorm

    def build(self, hp, inputs=None):
        input_node = tf.nest.flatten(inputs)[0]

        n_rg = utils.add_to_hp(self.n_rg, hp)

        if self.convs_optional:
            conv_type = hp.Choice(
                CONV_TYPE, [CONV2D, CONV2D_WEIGHTNORM], default=CONV2D_WEIGHTNORM)
            final_conv = hp.Boolean('final_conv')
        else:
            conv_type = CONV2D
            final_conv = True

        conv = self._get_conv(conv_type)
        filters = utils.add_to_hp(self.filters, hp)

        # params fixed for all res groups
        if not self.diversify_res:

            kernel_size = utils.add_to_hp(self.kernel_size, hp)
            n_rcab = utils.add_to_hp(self.n_rcab, hp)
            r_scaling = utils.add_to_hp(self.r_scaling, hp)

            x = x_1 = conv(filters=filters, kernel_size=kernel_size,
                           strides=1, padding='same')(input_node)
            x = rir(x, filters, kernel_size, conv, n_rg=n_rg,
                    n_rcab=n_rcab, r_scaling=r_scaling)
            x = conv(filters=filters, kernel_size=kernel_size,
                     strides=1, padding='same')(x)
            x = Add()([x_1, x])
            x = upsample(x, filters, self.scale)
            output_node = conv(filters=3, kernel_size=kernel_size,
                               strides=1, padding='same')(x)

        # each res block own params
        else:
            conv_kernel_size = utils.add_to_hp(
                self.kernel_size, hp, "conv_kernel_size")

            x = x_1 = conv(filters=filters, kernel_size=conv_kernel_size,
                           strides=1, padding='same')(input_node)

            # rir structure
            for i in range(n_rg):
                s = x

                for j in range(utils.add_to_hp(self.n_rcab, hp, "n_rcab_{i}".format(i=i))):
                    s_1 = x
                    x = rcab(x, filters, utils.add_to_hp(
                        self.kernel_size, hp, "kernel_size_{i}_{j}".format(i=i, j=j)), conv, scale=utils.add_to_hp(self.r_scaling, hp, "r_scaling_{i}_{j}".format(i=i, j=j)))

                x = conv(filters, utils.add_to_hp(
                    self.kernel_size, hp, "kernel_size_{i}".format(i=i)), strides=1, padding='same')(x)
                x = Add()([x, s_1])

            x = conv(filters=filters, kernel_size=conv_kernel_size,
                     strides=1, padding='same')(x)
            x = Add()([x, s])

            x = conv(filters=filters, kernel_size=conv_kernel_size,
                     strides=1, padding='same')(x)
            x = Add()([x_1, x])
            output_node = upsample(x, filters, self.scale)

            if final_conv:
                output_node = conv(filters=filters, kernel_size=conv_kernel_size,
                                   strides=1, padding='same')(x)

        return output_node
