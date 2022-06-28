# FROM https://github.com/hieubkset/Keras-Image-Super-Resolution

import sys

import tensorflow as tf
from tensorflow.keras.layers import (Activation, Add, Conv2D, Dense,
                                     GlobalAveragePooling2D, Input, Lambda,
                                     Multiply, Reshape)
from tensorflow.keras.models import Model

from utils import denormalize, normalize

sys.setrecursionlimit(10000)


def sub_pixel_conv2d(scale, **kwargs):
    return Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)


def upsample(input_tensor, filters, scale):
    x = Conv2D(filters=filters * scale * scale, kernel_size=3,
               strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv2d(scale)(x)
    x = Activation('relu')(x)
    return x


def ca(input_tensor, filters, reduce=16):
    x = GlobalAveragePooling2D()(input_tensor)
    x = Reshape((1, 1, filters))(x)
    x = Dense(filters/reduce,  activation='relu',
              kernel_initializer='he_normal', use_bias=False)(x)
    x = Dense(filters, activation='sigmoid',
              kernel_initializer='he_normal', use_bias=False)(x)
    x = Multiply()([x, input_tensor])
    return x


def rcab(input_tensor, filters, scale=0.1):
    x = Conv2D(filters=filters, kernel_size=3,
               strides=1, padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = ca(x, filters)
    if scale:
        x = Lambda(lambda t: t * scale)(x)
    x = Add()([x, input_tensor])

    return x


def rg(input_tensor, filters, n_rcab=20):
    x = input_tensor
    for _ in range(n_rcab):
        x = rcab(x, filters)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x, input_tensor])

    return x


def rir(input_tensor, filters, n_rg=10,n_rcab=20):
    x = input_tensor
    for _ in range(n_rg):
        x = rg(x, filters=filters,n_rcab=n_rcab)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x, input_tensor])

    return x


def generator(filters=64, scale=2,n_rg=10, n_res=20):
    inputs = Input(shape=(None, None, 3))
    x = Lambda(normalize)(inputs)
    x = x_1 = Conv2D(filters=filters, kernel_size=3,
                     strides=1, padding='same')(x)
    x = rir(x, filters=filters,n_rg=n_rg, n_rcab=n_res)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Add()([x_1, x])
    x = upsample(x, filters, scale)
    x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)
    x = Lambda(denormalize)(x)
    return Model(inputs=inputs, outputs=x)
