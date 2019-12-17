import numpy as np

from predictor import full_model_predictor as fmp
model_path = r"selected_models\model-06-0.0740.hdf5"


def create_model(model_path, height, width):
    height = height + 32
    width = width + 32
    if height % 8 != 0:
        height = height - height % 8 + 8
    if width % 8 != 0:
        width = width - width % 8 + 8
    model = fmp.get_conv_model(height=height, width=width)
    model.load_weights(model_path)
    return model


def get_prediction(model, obs):
    height = obs.shape[0] + 32
    width = obs.shape[1] + 32
    if height % 8 != 0:
        height = height - height % 8 + 8
    if width % 8 != 0:
        width = width - width % 8 + 8
    margin_h = (height - obs.shape[0]) // 2
    margin_w = (width - obs.shape[1]) // 2
    final_obs = np.zeros(shape=(height, width))
    final_obs[margin_h:margin_h + obs.shape[0], margin_w:margin_w + obs.shape[1]] = obs
    final_obs = np.reshape(final_obs, newshape=(1, final_obs.shape[0], final_obs.shape[1], 1))
    prediction, _ = model.predict(x=[final_obs, np.ones((1, height * width)).astype(float)])
    prediction = np.exp(prediction) / (np.exp(prediction) + 1.0)
    prediction = prediction.reshape([height,width])
    prediction = prediction[margin_h:-margin_h, margin_w:-margin_w]
    return prediction
