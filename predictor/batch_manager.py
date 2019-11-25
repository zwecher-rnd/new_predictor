import os

import pandas as pd
import numpy as np

from tensorflow.keras.utils import Sequence
import logging

from predictor import config

OBS_FNAMES = "obs_fnames"
LABELS_FNAMES = "labels_fnames"


class BatchManager:
    def __init__(self, metadata_df, test_num, shuffle=False):
        self.current_ind = 0
        if shuffle:
            metadata_df = metadata_df.iloc[np.random.permutation(len(metadata_df))]
            metadata_df.reset_index(inplace=True)
        self.train_metadata_df = metadata_df.iloc[:-test_num, :]
        self.test_metadata_df = metadata_df.iloc[-test_num:, :]
        self.examples_num = len(metadata_df)

    def get_next_train_batch(self, batch_size):
        batch_df = self.train_metadata_df.iloc[self.current_ind:self.current_ind + batch_size].copy()
        self.current_ind = (self.current_ind + batch_size)
        if self.current_ind > len(self.train_metadata_df):
            self.current_ind = 0
        return get_samples(batch_df)

    def get_test_batch(self):
        batch_df = self.test_metadata_df.copy()
        return get_samples(batch_df)


def create_metadata_df(obs_path=config.OBS_PATH, labels_path=config.LABELS_PATH, num_of_obs=config.NUM_OF_OBS):
    obs_fnames = pd.Series(os.listdir(obs_path)).sort_values()
    obs_fnames.index = obs_fnames.index // num_of_obs
    obs_fnames = config.OBS_PATH + os.path.sep + obs_fnames
    labels_fnames = pd.Series(os.listdir(labels_path)).sort_values()
    labels_fnames = config.LABELS_PATH + os.path.sep + labels_fnames
    metadata_df = pd.concat([obs_fnames, labels_fnames], axis=1)
    metadata_df.columns = [OBS_FNAMES, LABELS_FNAMES]
    return metadata_df


class BuildingsGenrator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        x_series = self.x.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        y_series = self.y.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = pd.concat([x_series, y_series], axis=1)
        batch_df.columns = [OBS_FNAMES, LABELS_FNAMES]
        # return get_samples(batch_df)
        return get_samples(batch_df)


def get_samples(batch_df):
    obs = []
    labels = []
    for _, row in batch_df.iterrows():
        new_obs = np.loadtxt(row[OBS_FNAMES])
        base_obs = np.zeros(np.array(new_obs.shape) + 2*config.PADDING)
        base_labels = np.zeros(np.array(new_obs.shape) + 2*config.PADDING)
        x1 = np.random.randint(-5, 6) + config.PADDING
        x2 = np.random.randint(-5, 6) + config.PADDING
        new_obs[new_obs == 0] = -1
        new_obs[new_obs == 3] = 0
        base_obs[x1:x1 + (config.HEIGHT - 2 * config.PADDING), x2:x2 + (config.WIDTH - 2 * config.PADDING)] = new_obs
        obs.append(base_obs)
        new_labels = np.loadtxt(row[LABELS_FNAMES])
        base_labels[x1:x1 + (config.HEIGHT - 2 * config.PADDING),
        x2:x2 + (config.WIDTH - 2 * config.PADDING)] = new_labels
        labels.append(base_labels)
    labels = np.array(labels)
    tmp_output = np.expand_dims(np.array(obs), 3), labels.reshape(labels.shape[0], -1)
    return tmp_output, tmp_output[1]
