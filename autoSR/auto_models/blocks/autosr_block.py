from typing import Optional, List,Dict

import autokeras as ak
import tensorflow as tf
from keras_tuner.engine import hyperparameters

from .blocks import Denormalize01Block, Normalize01Block
from .rcan_block import RCANBlock
from .wdsr_block import WDSRBlock

MODEL_TYPE = "model_type"
WDSR = "wdsr"
RCAN = "rcan"

class AutoSR_Block(ak.Block):
    """ Block for super resolution for remote sensing. Searches mutations of either WDSR or RCAN 
        and uses pretrained weights of remote sensing datasets.
        
        Inputs:
        version: str                        which version of the block, currently only v1 exists
        pretrained_weights: List            list of datasets/pretrained weights that is allowed
        model_type: str                     either RCAN or WDSR, can be set to restrict search space
        scale: int                          super resolution scaling factor
        pretrained_weights_dir: dictionary  dictionary containing paths to pretrained weights, of shape {method:{dataset:path}}

        outputs:
        AutoKeras block
    """

    def __init__(
        self, 
        version: str, 
        pretrained_weights: List[str],
        model_type: Optional[str] = None, 
        scale: Optional[int] = 2, 
        pretrained_weights_dir: Optional[Dict] = {"wdsr":{
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
                            }},
        **kwargs
    ):

        super().__init__(**kwargs)
        self.pretrained_weights_dir=pretrained_weights_dir
        self.scale = scale
        self.version = version
        self.pretrained_weights = pretrained_weights

        if model_type is not None and model_type != "wdsr" and model_type != "rcan":
            raise Exception(f"invalid model_type {model_type}")

        self.model_type = model_type


    def get_config(self):
        config = super().get_config()
        config.update({"version": self.version,
                      "scale": self.scale, "model_type": self.model_type,"pretrained_weights":self.pretrained_weights,"pretrained_weights_dir":self.pretrained_weights_dir})
        return config

    @classmethod
    def from_config(cls, config):
        config["version"] = hyperparameters.deserialize(config["version"])
        config["scale"] = hyperparameters.deserialize(config["scale"])
        config["pretrained_weights"] = hyperparameters.deserialize(config["pretrained_weights"])
        config["pretrained_weights_dir"]=hyperparameters.deserialize(config["pretrained_weights_dir"])
        return cls(**config)

    def _build_model(self, hp,pretrained_weights, output_node, model_type):
        if model_type == WDSR:
            return WDSRBlock(self.pretrained_weights_dir,scale=self.scale, pretrained_on_dataset=pretrained_weights).build(hp, inputs=output_node)
        elif model_type == RCAN:
            return RCANBlock(self.pretrained_weights_dir,scale=self.scale,n_rg=1, pretrained_on_dataset=pretrained_weights).build(hp, inputs=output_node)  


    def build(self, hp, inputs=None):
        inputs = tf.nest.flatten(inputs)[0]
        weights_options=hp.Choice("pretrained_weights", self.pretrained_weights)
        # normalize
        output_node = Normalize01Block().build(hp, inputs)

        # logic block
        if self.model_type is None:
            model_type = hp.Choice(MODEL_TYPE, [WDSR, RCAN])
            with hp.conditional_scope(MODEL_TYPE, [model_type]):
                output_node = self._build_model(hp, weights_options,output_node, model_type)
        else:
            output_node = self._build_model(hp,  weights_options,output_node,self.model_type)

        # denormalize
        output_node = Denormalize01Block().build(hp, output_node)
        return output_node
