
import tensorflow as tf
from utils import resolve


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]
