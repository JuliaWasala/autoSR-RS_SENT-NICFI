
from typing import List,Optional
import autokeras as ak
import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv2D,Lambda, Input
from tensorflow.python.keras.models import Model
from keras_tuner.engine import hyperparameters
from baselines.wdsr.wdsr import res_block_b
from baselines.rcan.rcan import rcab
import pickle
from functools import partial

RB="RB"
RCAB="RCAB"

PRETRAINED_WEIGHTS = {"wdsr":{
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
                            }}

class WDSRResBBlock(ak.Block):
    """ Block implementing Resblock-B from WDSR
    inputs:
        pretrained_weights: np array
    outputs:
        tf tensor"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        config=super().get_config()
        config.update(
            {
            
            }
        )
        return config

    @classmethod
    def from_config(cls,config):
        return cls(**config)

    def build(self,hp, inputs=None):
        num_filters=32
        kernel_size=3
        expansion=6
        linear=0.8
        scaling=None

        input_node=tf.nest.flatten(inputs)[0]
        output_node = res_block_b(input_node, num_filters, expansion, kernel_size, scaling)

        # model=Model(input_node, output_node)
        # print("wdsr resblock")
        # print("model.summary")
        # model.load_weights(self.pretrained_weights)
        return output_node


class RCANRCABBlock(ak.Block):
    """ Block implementing RCAB from RCAN
    inputs:
        pretrained_weights: np array
    outputs:
        tf tensor"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        config=super().get_config()
        config.update(
            {
            }
        )
        return config

    @classmethod
    def from_config(cls,config):
        config["pretrained_weights"]=hyperparameters.deserialize(config["pretrained_weights"])
        return cls(**config)

    def build(self,hp, inputs=None):
        num_filters=64

        input_node=tf.nest.flatten(inputs)[0]
        output_node = rcab(input_node, num_filters)

        # model=Model(input_node, output_node)
        # print("rcan rcab")
        # print("model.summary")
        # model.load_weights(self.pretrained_weights)
        return output_node

class RCANRGBlock(ak.Block):
    """ Block implementing RG from RCAN
    inputs:
        pretrained_weights: cerrado | sr_ucmerced | sr_so2sat | oli2msi        pretrained weights
        residual_block_type: RB | RCAB  type of res block (RB= WDSR, RCAB = RCAN)
        num_res_blocks: int             number of res blocks in block
    outputs:
        tf tensor"""
    def __init__(self, pretrained_dataset: List,residual_block_type:str,num_res_blocks:int, rg_id: Optional[int]=0,**kwargs):
        super().__init__(**kwargs)
        self.pretrained_dataset=pretrained_dataset
        self.residual_block_type=residual_block_type
        self.num_res_blocks=self.num_res_blocks
        self.rg_id=rg_id

        ########################
        # get pretrained weights
        #########################

        if self.residual_block_type==RB:
            # get weights of resblocks
            with open(f"{PRETRAINED_WEIGHTS['wdsr'][self.pretrained_dataset]}/resblocks.pkl", "rb") as f:
                pretrained_resblock_weights=pickle.load(f)
            if self.num_res_blocks < len(pretrained_resblock_weights):
                resblock_weights=pretrained_resblock_weights[:self.num_res_blocks]
            else:
                num_diff=self.num_res_blocks - len(pretrained_resblock_weights)
                resblock_weights=pretrained_resblock_weights
                resblock_weights.extend([pretrained_resblock_weights[-1]*num_diff])

            # get last two layers of rg
            with open(f"{PRETRAINED_WEIGHTS['rcan'][self.pretrained_dataset]}/rg.pkl", "rb") as f:
                all_rg_weights=pickle.load(f)
            self.weights=[layer for sublist in resblock_weights for layer in sublist ]
            self.weights.extend(all_rg_weights[self.rg_id][200:])

        elif self.residual_block_type==RCAB:
            resblock_weights=[]
            with open(f"{PRETRAINED_WEIGHTS['rcan'][self.pretrained_dataset]}/rg.pkl", "rb") as f:
                all_rg_weights=pickle.load(f)
            rg_weights=all_rg_weights[self.rg_id]
            
            if self.num_res_blocks < (len(rg_weights)-2)//10:
                self.weights=rg_weights[:self.num_res_blocks*10]+rg_weights[:-2]
            else:
                num_diff=self.num_res_blocks - ((len(rg_weights)-2)//10)
                self.weights=rg_weights[:-2]
                self.weights.extend([rg_weights[-12:-2]]*num_diff)
                self.weights=self.weights+rg_weights[-2:]


    def get_config(self):
        config=super().get_config()
        config.update(
            {
                "pretrained_dataset":self.pretrained_dataset,
                "residual_block_type":self.residual_block_type,
                "num_res_blocks":self.num_res_blocks,
            }
        )
        return config

    @classmethod
    def from_config(cls,config):
        config["pretrained_dataset"]=hyperparameters.deserialize(config["pretrained_dataset"])
        config["residual_block_type"]=hyperparameters.deserialize(config["residual_block_type"])
        config["num_res_blocks"]=hyperparameters.deserialize(config["num_res_blocks"])
        return cls(**config)

    @staticmethod
    def _get_resblock(resblock_type):
        if resblock_type == RB:
            return partial(res_block_b,num_filters=32, expansion=6, kernel_size=3, scaling=None)

        elif resblock_type==RCAB:
            return partial(rcab, filters=64)

    def build(self,hp, inputs=None):
        resblock= self._get_resblock(self.residual_block_type)

        input_node=tf.nest.flatten(inputs)[0]
        output_node=input_node
    
        for _ in range(self.num_res_blocks):
            output_node=resblock(output_node)

        output_node=Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        output_node=Add()([input_node, output_node])
        model=Model(input_node, output_node)
        print("RG")
        print("model.summary")
        model.load_weights(self.weights)
        return model