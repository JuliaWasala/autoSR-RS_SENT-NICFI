from queue import Full
from typing import Optional

import autokeras as ak
import tensorflow as tf
from keras_tuner.engine import hyperparameters

from .blocks import Denormalize01Block, Normalize01Block
from .rcan_block import RCANBlock
from .wdsr_block import WDSRBlock
from .fully_custom_block import FullyCustomBlock

MODEL_TYPE = "model_type"
WDSR = "wdsr"
RCAN = "rcan"
FULLY_CUSTOM = "fully_custom"


class AutoSR_Block(ak.Block):
    """ Block implementing multiple search space versions for AutoSR-RS
    There are 4 options: a | b | c | d:
        a:              either WDSR / RCAN. Model can be mutated: depth (num res blocks),
                        num of filters, kernel size and scaling are shared. 

        b:              either WDSR / RCAN. Model can be mutated: each res block has its own parameters.

        c:              In addition to b, now the convolutions can be either Conv2D or Conv2D weightnorm. 
                        convolutions that are not in both RCAN / WDSR can be left out

        d:              Fully custom. Generalized res block. 

    """

    def __init__(
        self, version: str, model_type: Optional[str] = None, scale: Optional[int] = 2, **kwargs
    ):
        super().__init__(**kwargs)
        self.scale = scale
        self.version = version

        if model_type is not None and model_type != "wdsr" and model_type != "rcan":
            raise Exception(f"invalid model_type {model_type}")

        self.model_type = model_type

        if self.version == "a":
            self.diversify_res = False
            self.convs_optional = False
        elif self.version == "b":
            self.diversify_res = True
            self.convs_optional = False
        elif self.version == "c":
            self.diversify_res = True
            self.convs_optional = True
        elif self.version == "d":
            self.diversify_res = None
            self.convs_optional = None
            self.model_type = "fully_custom"

    def get_config(self):
        config = super().get_config()
        config.update({"version": self.version,
                      "scale": self.scale, "model_type": self.model_type})
        return config

    @classmethod
    def from_config(cls, config):
        config["version"] = hyperparameters.deserialize(config["version"])
        config["scale"] = hyperparameters.deserialize(config["scale"])
        return cls(**config)

    def _build_model(self, hp, output_node, model_type):
        if model_type == WDSR:
            return WDSRBlock(scale=self.scale, diversify_res=self.diversify_res, convs_optional=self.convs_optional).build(hp, inputs=output_node)
        elif model_type == RCAN:
            return RCANBlock(scale=self.scale, diversify_res=self.diversify_res, convs_optional=self.convs_optional).build(hp, inputs=output_node)
        elif model_type == FULLY_CUSTOM:
            return FullyCustomBlock(scale=self.scale).build(hp, inputs=output_node)

    def build(self, hp, inputs=None):
        inputs = tf.nest.flatten(inputs)[0]

        # normalize
        output_node = Normalize01Block().build(hp, inputs)

        # logic block
        if self.model_type is None:
            model_type = hp.Choice(MODEL_TYPE, [WDSR, RCAN])
            with hp.conditional_scope(MODEL_TYPE, [model_type]):
                output_node = self._build_model(hp, output_node, model_type)
        else:
            output_node = self._build_model(hp, output_node, self.model_type)

        # denormalize
        output_node = Denormalize01Block().build(hp, output_node)
        return output_node
