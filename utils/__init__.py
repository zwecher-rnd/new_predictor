import os
import sys
import logging

from tensorflow.keras import backend as K


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
    outputs = [layer.output for layer in model.layers if
               layer.name in layers_name]  # all layer outputs except first (input) layer
    functor = K.function([inp], outputs)
    return functor
