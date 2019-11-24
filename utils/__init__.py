import os
import sys
import logging

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Conv2D, Flatten, Multiply
import numpy as np


def set_log(level=logging.INFO, filename="info", dir=r".."):
    fmt = '\n%(asctime)s [%(levelname)s: %(filename)s %(lineno)d] %(message)s'
    log_path = os.path.join(dir, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    filename = filename
    logging.basicConfig(
        level=level, format=fmt,
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(log_path, filename), mode='w'),
            logging.StreamHandler(stream=sys.stdout)]
    )
    logging.info("path: {}".format(log_path))


def get_inermediate_layers_func(model, layers_name):
    inp = model.input  # input placeholder
    outputs = []
    for layer_name in layers_name:
        outputs.append(model.get_layer(layer_name).output)  # all layer outputs except first (input) layer
    functor = K.function([inp], outputs)
    return functor


def get_inermediate_layers_val(func, input_batch):
    return func(input_batch)


def logits_to_probs(binary_logits_batch):
    return np.exp(binary_logits_batch) / (1.0 + np.exp(binary_logits_batch))


def resize_image_layer(shape, name):
    q = Lambda(lambda x: Flatten()(tf.image.resize(x, shape, method=tf.image.ResizeMethod.BICUBIC)),
               name=name)
    return q


def get_mask_layer(name):
    def get_mask_func(x):
        mask = tf.cast(tf.not_equal(x, 0.0), tf.float32)
        proximity_mask = Conv2D(kernel_size=10, filters=1, padding="same", kernel_initializer="ones", )(mask)
        proximity_mask2 = tf.keras.backend.greater(proximity_mask, 0.0)
        proximity_mask3 = Flatten()(proximity_mask2)
        return tf.cast(proximity_mask3, tf.float32)

    return Lambda(lambda x: get_mask_func(x), name=name, trainable=False)


def apply_mask_layer(name):
    def get_apply_func(x):
        masked_loss_tmp = Multiply(name="masked_loss")([x[0], x[1]])
        masked_loss = tf.boolean_mask(masked_loss_tmp, tf.cast(masked_loss_tmp > 0, tf.bool))
        return masked_loss

    return Lambda(lambda x: get_apply_func(x), name=name, trainable=False)


def sigmoid_cross_entropy_with_logits_layer(name):
    return Lambda(lambda x: tf.nn.sigmoid_cross_entropy_with_logits(labels=x[0], logits=x[1]),
                  name=name)
