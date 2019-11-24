import logging

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization

from predictor import config, batch_manager

from utils import get_inermediate_layers_func, resize_image_layer


def loss(y_true, y_pred):
    # logging.info(labels)

    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))


def get_model():
    ao = optimizers.Adam(learning_rate=0.001)

    model = Sequential()
    model.add(
        Conv2D(filters=15, kernel_size=11, activation="relu", input_shape=(config.HEIGHT, config.WIDTH, 1),
               name="my_input"))
    model.add(MaxPool2D(pool_size=4, strides=1))
    model.add(Conv2D(filters=15, kernel_size=11, activation="relu", name="conv1"))
    model.add(MaxPool2D(pool_size=4, strides=1))
    model.add(Conv2D(filters=15, kernel_size=11, activation="relu", name="conv2"))
    model.add(Conv2D(filters=1, kernel_size=11, name="conv3"))
    # model.add(resize_image((config.HEIGHT, config.WIDTH)))
    model.add(Flatten(name="flatten"))
    model.add(Dense(config.HEIGHT * config.WIDTH, name="dense"))
    model.compile(optimizer=ao, loss=loss)
    model.summary()
    return model


def train_model(model, epochs=5):
    metadata_df = batch_manager.create_metadata_df()
    train_metadata_df = metadata_df.iloc[:-256]
    test_metadata_df = metadata_df.iloc[-256:]
    # bm = batch_manager.BatchManager(metadata_df, 100)
    dg = batch_manager.BuildingsGenrator(x_set=train_metadata_df.iloc[:, 0], y_set=train_metadata_df.iloc[:, 1],
                                         batch_size=256)
    validation_data = batch_manager.get_samples(test_metadata_df)
    model.fit_generator(generator=dg, epochs=epochs, verbose=1, validation_data=validation_data)

# il = get_inermediate_layers_func(model, ["conv2d"])
# for i in range(num_of_iterations):
#     obs_batch, labels_batch = bm.get_next_train_batch(batch_size=256)
#     c1 = model.train_on_batch(obs_batch, labels_batch)
#     logging.info(c1)
#     logging.info(il([obs_batch])[1])
