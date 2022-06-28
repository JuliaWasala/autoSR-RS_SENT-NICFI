from typing import Optional, Union

import autokeras as ak
import tensorflow as tf
from autokeras.utils import utils
from keras_tuner.engine import hyperparameters
from tensorflow.python.keras.layers import Activation, Add, Conv2D, Lambda

from .rcan_block import ca, sub_pixel_conv2d
from .wdsr_block import conv2d_weightnorm

CONV_TYPE = "conv_type"
CONV2D = "conv2d"
CONV2D_WEIGHTNORM = "conv2d_weightnorm"


def resblock(input, filters, kernel_size, conv, has_ca, has_linear, expansion, scaling):
    """ Residual block generalizing the RCAN channel attention block and WDSR res block B
    inputs:
        input: tensor           input
        filters: int            number of filters
        kernel_size: int        number
        conv: keras layer       type of conv, either Conv2D or
                                Conv2D_weightnorm
        has_ca: bool            whether block as channel attention module
        has_linear: bool        whether block has layer with linearly scaled
                                num of filters
        expansion: bool         filter expansion in expansion layer
        scaling: int            scaling factor

    return:
        tensor
    """

    linear = 0.8

    if expansion:
        x = conv(filters*6, 1, padding='same', activation='relu')(input)
    else:
        x = conv(filters, kernel_size, padding='same',
                 activation='relu')(input)

    if has_linear:
        x = conv(int(filters*linear), 1, padding='same')(x)

    x = conv(filters, kernel_size, padding='same')(x)

    if has_ca:
        x = ca(x, filters)

    if scaling != 1:
        x = Lambda(lambda t: t * scaling)(x)

    x = Add()([x, input])
    return x


def rg(input, filters, kernel_size, conv, has_ca, has_linear, expansion, n_res, scaling):
    """ Residual group based off RCAN
    inputs:
        input: tensor           input
        filters: int            number of filters
        kernel_size: int        number
        conv: keras layer       type of conv, either Conv2D or
                                Conv2D_weightnorm
        has_ca: bool            whether block as channel attention module
        has_linear: bool        whether block has layer with linearly scaled
                                num of filters
        expansion: bool         filter expansion in expansion layer
        n_res: int              number of residual blocks
        scaling: int            scaling factor

    return:
        tensor
    """
    x = input
    for _ in range(n_res):
        x = resblock(x, filters, kernel_size, conv,
                     has_ca, has_linear, expansion, scaling)
    x = conv(filters, kernel_size, strides=1, padding='same')(x)
    x = Add()([x, input])

    return x


def rir(input, filters, kernel_size, conv, has_ca, has_linear, expansion, n_rg, n_res, scaling):
    """ Residual-in-residual block based on RCAN
    inputs:
        input: tensor           input
        filters: int            number of filters
        kernel_size: int        number
        conv: keras layer       type of conv, either Conv2D or
                                Conv2D_weightnorm
        has_ca: bool            whether block as channel attention module
        has_linear: bool        whether block has layer with linearly scaled
                                num of filters
        expansion: bool         filter expansion in expansion layer
        n_rg: int               number of residual groups
        n_res: int              number of residual blocks
        scaling: int            scaling factor

    return:
        tensor
    """

    x = input
    for i in range(n_rg):
        x = rg(x, filters, kernel_size, conv, has_ca,
               has_linear, expansion, n_res, scaling)
    x = conv(filters=filters, kernel_size=kernel_size,
             strides=1, padding='same')(x)
    x = Add()([x, input])

    return x


def upsample(input, filters, kernel_size, scale, activation):
    """ Upsample block generalizing RCAN & WDSR
    inputs:
        input: tensor       input tensor
        filters: int        number of filters
        kernel_size: int    kernel_size
        activation: bool    whether to add activation or not
    return:
        output tensor
    """
    x = Conv2D(filters=filters * scale * scale, kernel_size=kernel_size,
               strides=1, padding='same')(input)
    x = sub_pixel_conv2d(scale)(x)
    if activation:
        x = Activation('relu')(x)
    return x


class FullyCustomBlock(ak.Block):
    """ Block implementing fully customized search space for AutoSR-RS
    inputs:
    scale: int=2                            upscaling factor of datset
    filters:     int |                      number of filters
                 hyperparameters.Choice
    kernel_size: int |                      kernel size
                 hyperparameters.Choice
    scaling:     float |                    residual channel attention block scaling
                 hyperparameters.Float
    n_res:       int |                      number of residual channel attention 
                 hyperparameters.Int        blocks per residual group
    wide_in:     bool |                     whether to have wide activation (activation 
                 hyperparameters.Boolean    before (False) or after (True) split)
    wide_out:    bool |                     whether to have wide activation (activation 
                 hyperparameters.Boolean    before (False) or after (True) merge)
    diversify_res: bool = False             whether each res group & channel attention
                                            block has own set of parameters (True)
                                            or shared parameters (False) 
    convs_optional: bool = False            whether first & last conv are optional AND
                                            whether Conv2D can be swapped for
                                            weight normalized Conv2D
    """

    def __init__(self,
                 scale: Optional[int] = 2,
                 filters: Optional[Union[int, hyperparameters.Int]] = None,
                 kernel_size: Optional[Union[int, hyperparameters.Int]] = None,
                 scaling: Optional[Union[float, hyperparameters.Float]] = None,
                 n_res: Optional[Union[int, hyperparameters.Int]] = None,
                 wide_in: Optional[Union[bool, hyperparameters.Choice]] = None,
                 wide_out: Optional[Union[bool, hyperparameters.Choice]] = None,
                 rir: Optional[Union[bool, hyperparameters.Choice]] = None,
                 merge_before_upsample: Optional[Union[bool, hyperparameters.Choice]] = None,
                 expansion: Optional[bool] = None,
                 has_linear: Optional[bool] = None,
                 has_ca: Optional[bool] = None,
                 upsample_activation: Optional[bool] = None,
                 **kwargs,):

        super().__init__(**kwargs)
        self.scale = scale
        self.filters = utils.get_hyperparameter(
            filters,
            hyperparameters.Choice(
                "filters", [16, 32, 64, 128, 256, 512], default=64),
            int,
        )
        self.kernel_size = utils.get_hyperparameter(
            kernel_size,
            hyperparameters.Choice("kernel_size", [3, 5, 7, 9], default=3),
            int,
        )
        self.scaling = utils.get_hyperparameter(
            scaling,
            hyperparameters.Float("scaling", 0.1, 1,
                                  step=0.1, default=1),
            int,
        )
        self.n_res = utils.get_hyperparameter(
            n_res,
            hyperparameters.Int("n_res", 4, 40, step=2, default=32),
            int,
        )
        self.wide_in = utils.get_hyperparameter(
            wide_in,
            hyperparameters.Boolean("wide_in"),
            bool,
        )
        self.rir = utils.get_hyperparameter(
            rir,
            hyperparameters.Boolean("rir"),
            bool,
        )
        self.merge_before_upsample = utils.get_hyperparameter(
            merge_before_upsample,
            hyperparameters.Boolean("merge_before_upsample"),
            bool,
        )
        self.expansion = expansion
        self.has_linear = has_linear
        self.has_ca = has_ca
        self.upsample_activation = upsample_activation

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters,
                       "kernel_size": self.kernel_size,
                       "scaling": self.scaling,
                       "n_res": self.n_res,
                       "scale": self.scale,
                       "wide_in": self.wide_in,
                       "rir": self.rir,
                       "merge_before_upsample": self.merge_before_upsample,
                       "expansion": self.expansion,
                       "has_linear": self.has_linear,
                       "has_ca": self.has_ca,
                       "upsample_activation": self.upsample_activation,
                       })
        return config

    @classmethod
    def from_config(cls, config):
        config["kernel_size"] = hyperparameters.deserialize(
            config["kernel_size"])
        config["filters"] = hyperparameters.deserialize(config["filters"])
        config["n_res"] = hyperparameters.deserialize(config["n_res"])
        config["scaling"] = hyperparameters.deserialize(config["scaling"])
        config["scale"] = hyperparameters.deserialize(config["scale"])
        config["wide_in"] = hyperparameters.deserialize(config["wide_in"])
        config["rir"] = hyperparameters.deserialize(config["rir"])
        config["merge_before_upsample"] = hyperparameters.deserialize(
            config["merge_before_upsample"])
        config["expansion"] = hyperparameters.deserialize(config["expansion"])
        config["has_linear"] = hyperparameters.deserialize(
            config["has_linear"])
        config["has_ca"] = hyperparameters.deserialize(config["has_ca"])
        config["upsample_activation"] = hyperparameters.deserialize(
            config["upsample_activation"])
        return cls(**config)

    @staticmethod
    def _get_conv(conv_type):
        if conv_type == CONV2D:
            return Conv2D
        elif conv_type == CONV2D_WEIGHTNORM:
            return conv2d_weightnorm

    def build(self, hp, inputs=None):
        inputs = tf.nest.flatten(inputs)[0]

        filters = utils.add_to_hp(self.filters, hp)
        kernel_size = utils.add_to_hp(self.kernel_size, hp)
        n_res = utils.add_to_hp(self.n_res, hp)
        scaling = utils.add_to_hp(self.scaling, hp)
        wide_in = utils.add_to_hp(self.wide_in, hp)
        rir = utils.add_to_hp(self.rir, hp)
        merge_before_upsample = utils.add_to_hp(self.merge_before_upsample, hp)
        conv_type = hp.Choice(CONV_TYPE, [CONV2D, CONV2D_WEIGHTNORM])
        conv = self._get_conv(conv_type)

        expansion = self.expansion
        if expansion is None:
            expansion = hp.Boolean("expansion")

        has_linear = self.has_linear
        if has_linear is None:
            has_linear = hp.Boolean("has_linear")

        has_ca = self.has_ca
        if has_ca is None:
            has_ca = hp.Boolean("has_ca")

        upsample_activation = self.upsample_activation
        if upsample_activation is None:
            upsample_activation = hp.Boolean("upsample_activation")

        # conv before or after split
        # x = main branch
        # x_1 = skip branch
        if not wide_in:
            x = x_1 = conv(filters, kernel_size, padding='same')(inputs)
        else:
            x = conv(filters, kernel_size, padding='same')(inputs)
            x_1 = inputs

        # optional RIR
        if rir:
            n_rg = hp.Int("n_rg", 2, 16, step=2, default=10)
            x = rir(x, filters, kernel_size, conv, has_ca,
                    has_linear, expansion, n_rg, n_res, scaling)

        else:
            for _ in range(n_res):
                x = resblock(x, filters, kernel_size, conv,
                             has_ca, has_linear, expansion, scaling)

        if hp.Boolean("optional_conv"):
            x = conv(filters, kernel_size, padding='same')(x)

        # merge and upsample
        if merge_before_upsample:
            x = Add()([x_1, x])
            x = upsample(x, filters, kernel_size,
                         self.scale, upsample_activation)

        else:
            x = upsample(x, filters, kernel_size,
                         self.scale, upsample_activation)
            x_1 = upsample(x_1, filters, kernel_size,
                           self.scale, upsample_activation)
            x = Add()([x_1, x])

        if hp.Boolean("final_conv"):
            x = conv(filters, kernel_size, padding="same")(x)

        return x
