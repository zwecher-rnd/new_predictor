import logging

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten

from predictor import config, batch_manager


def loss(labels, logits):
    logging.info(labels)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def get_model_old():
    ao = optimizers.Adam(learning_rate=0.001)
    obs = Input(shape=(config.HEIGHT, config.WIDTH, 1), name="obs")
    # labels = Input(shape=(config.HEIGHT * config.WIDTH,), name="labels")
    first_conv = Conv2D(filters=7, kernel_size=7, activation="relu", name="first_conv")(obs)
    max_pool_2d = MaxPool2D(pool_size=4, strides=2, name="max_pool_2d")(first_conv)
    second_conv = Conv2D(filters=7, kernel_size=7, activation="relu", name="second_conv")(max_pool_2d)
    flatten = Flatten(name="flatten")(second_conv)
    dense = Dense(units=config.HEIGHT * config.WIDTH)(flatten)
    probs = tf.math.divide(tf.math.exp(dense), tf.reduce_sum(tf.math.exp(dense)), name="probs")
    model = Model(inputs=[obs], outputs=[dense])
    model.compile(optimizer=ao, loss=loss, outputs=[dense, probs])
    model.summary()
    return model


def get_model():
    ao = optimizers.Adam(learning_rate=0.001)
    obs = Input(shape=(config.HEIGHT, config.WIDTH, 1), name="obs")
    first_conv = Conv2D(filters=7, kernel_size=7, activation="relu", name="first_conv")(obs)
    max_pool_2d = MaxPool2D(pool_size=4, strides=2, name="max_pool_2d")(first_conv)
    second_conv = Conv2D(filters=7, kernel_size=7, activation="relu", name="second_conv")(max_pool_2d)
    flatten = Flatten(name="flatten")(second_conv)
    dense = Dense(units=config.HEIGHT * config.WIDTH)(flatten)
    #probs = tf.math.divide(tf.math.exp(dense), tf.reduce_sum(tf.math.exp(dense)), name="probs")
    model = Model(inputs=[obs], outputs=[dense])
    model.compile(optimizer=ao, loss=loss)
    model.summary()
    return model


