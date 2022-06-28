from typing import Optional

import autokeras as ak
import tensorflow as tf
from tensorflow.python.keras.layers import Activation, Conv2D, Lambda
from utils import denormalize, normalize


def sub_pixel_conv2d(scale, **kwargs):
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)


class UpscalingPixelShuffleBlock(ak.Block):
    def build(self, hp, scale=2, filters=64, inputs=None):
        # Get the input_node from inputs.
        # TODO check sota for num of filters
        # check kernel sizes
        input_node = tf.nest.flatten(inputs)[0]

        x = Conv2D(filters=filters * scale * scale, kernel_size=3,
                   strides=1, padding='same')(input)
        x = sub_pixel_conv2d(scale)(x)
        return x


class UpscalingInterpolationBlock(ak.Block):
    def build(self, hp, scale=2, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        layer = tf.keras.layers.UpSampling2D(
            size=(scale, scale), data_format=None, interpolation=hp.Choice("interpolation_alg", ["nearest", "bilinear"]))
        output_node = layer(input_node)
        return output_node


class UpscalingBlock(ak.Block):
    def __init__(
        self, scale:Optional[int]=2, upscale_type:Optional[str]=None,activation:Optional[bool]=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.scale = scale
        self.activation = activation
        self.upscale_type=upscale_type

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale,"upscale_type":self.upscale_type, "activation": self.activation})
        return config

    def build(self, hp, inputs=None):
        if not self.upscale_type:
            if hp.Choice("upscale_type", ["interpolation", "pixel_shuffle"]) == "interpolation":
                with hp.conditional_scope("upscale_type", ["interpolation"]):
                    outputs = UpscalingInterpolationBlock().build(hp, self.scale, inputs)
            else:
                with hp.conditional_scope("upscale_type", ["interpolation"]):
                    outputs = UpscalingPixelShuffleBlock().build(hp, self.scale, inputs)
        else: 
            if self.upscale_type == "interpolation":
                outputs = UpscalingInterpolationBlock().build(hp, self.scale, inputs)
            else:
                outputs = UpscalingPixelShuffleBlock().build(hp, self.scale, inputs)            
        if self.activation:
            outputs = Activation('relu')(outputs)
        return outputs


class Normalize01Block(ak.Block):
    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        output_node = Lambda(normalize)(input_node)
        return output_node


class Denormalize01Block(ak.Block):
    def build(self, hp, inputs=None):
        # Get the input_node from inputs.
        input_node = tf.nest.flatten(inputs)[0]
        output_node = Lambda(denormalize)(input_node)
        return output_node
