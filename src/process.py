import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from zntrack import Node, dvc, zn


class DataPreprocessor(Node):
    """Prepare kaggle dataset for training

    * normalize and reshape the features
    * one-hot encode the labels
    """

    # dependencies and parameters
    data: pathlib.Path = dvc.deps()
    dataset = zn.params()
    # outputs
    features: np.ndarray = zn.outs()
    labels: np.ndarray = zn.outs()

    def run(self):
        """Primary Node Method"""
        df = pd.read_csv((self.data / self.dataset / self.dataset).with_suffix(".csv"))

        self.labels = df.values[:, 0]
        self.labels = to_categorical(self.labels)
        self.features = df.values[:, 1:]

        self.normalize_and_scale_data()

    def normalize_and_scale_data(self):
        self.features = self.features / 255
        self.features = self.features.reshape((-1, 28, 28, 1))

    def plot_image(self, index):
        plt.imshow(self.features[index])
        plt.title(f"Label {self.labels[index].argmax()}")
        plt.show()
