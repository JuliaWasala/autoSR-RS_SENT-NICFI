import tensorflow as tf


def crop(image):
    margin = 4
    image = image[:, margin:-margin, margin:-margin]
    return tf.expand_dims(image, -1)


def psnr(hr, sr):
    # hr, sr = un_normalize(hr, sr)
    # hr = rgb_to_y(hr)
    # sr = rgb_to_y(sr)
    hr = crop(hr)
    sr = crop(sr)
    return tf.image.psnr(hr, sr, max_val=255)
