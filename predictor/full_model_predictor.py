import logging

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Multiply, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from predictor import config, batch_manager
import utils


def loss(y_true, y_pred):
    nums = tf.reduce_sum(tf.cast(y_pred > 0, tf.float32))
    sums = tf.reduce_sum(y_pred)
    return tf.divide(sums, nums)


def get_model(conv_layers_num=7, kernel_size=3):
    ao = optimizers.Adam(learning_rate=0.001)
    logging.info(config.HEIGHT)
    obs = Input(shape=(config.HEIGHT, config.WIDTH, 1), name="obs")
    labels = Input(shape=(config.HEIGHT * config.WIDTH), name="lab")
    inp = obs
    for l in range(conv_layers_num):
        conv = Conv2D(filters=25, kernel_size=kernel_size, activation="relu", name=f"conv_{l}")(inp)
        if l ==  conv_layers_num//2:
            inp = MaxPool2D(pool_size=2, strides=1, name=f"max_pool_2d_{l}")(conv)
        else:
            inp = conv

    last_conv = Conv2D(filters=1, kernel_size=7, name="last_conv")(inp)
    flat = Flatten(name = "flat")(last_conv)
    dense = Dense(config.HEIGHT*config.WIDTH, name="dense")(flat)
    #reshaped = Reshape(target_shape=(config.HEIGHT - 20, config.WIDTH - 20, 1), name="reshaped")(dense)

    logits = utils.resize_image_layer(shape=[config.HEIGHT, config.WIDTH,1], name="logits")(dense)
    loss_for_logits = utils.sigmoid_cross_entropy_with_logits_layer(name="loss_for_logits")([labels, logits])
    mask = utils.get_mask_layer(name="mask")(obs)
    masked_loss = Multiply(name="masked_loss")([loss_for_logits, mask])
    model = Model(inputs=[obs, labels], outputs=[masked_loss])
    model.compile(optimizer=ao, loss=loss)
    model.summary()
    return model


def train_model(model, epochs=5, batch_size=64):
    metadata_df = batch_manager.create_metadata_df()
    train_metadata_df = metadata_df.iloc[:-batch_size]
    test_metadata_df = metadata_df.iloc[-batch_size:]
    # bm = batch_manager.BatchManager(metadata_df, 100)
    dg = batch_manager.BuildingsGenrator(x_set=train_metadata_df.iloc[:, 0], y_set=train_metadata_df.iloc[:, 1],
                                         batch_size=batch_size)
    validation_data = batch_manager.get_samples(test_metadata_df)
    filepath = r"C:\Users\zwecher\Documents\mygit\new_predictor\models\models_6260\model-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit_generator(generator=dg, epochs=epochs, verbose=1, validation_data=validation_data,
                        callbacks=callbacks_list)
