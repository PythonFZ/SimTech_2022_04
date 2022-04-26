from zntrack import config
import pathlib
import kaggle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from zntrack import Node, NodeConfig, dvc, nodify, utils, zn
from zntrack.core import ZnTrackOption


@nodify(
    outs="dataset",
    params={"dataset": "datamunge/sign-language-mnist"}
)
def download_kaggle(cfg: NodeConfig):
    """Download dataset from kaggle"""
    kaggle.api.dataset_download_files(
        dataset=cfg.params.dataset, path=cfg.outs, unzip=True
    )


class DataPreprocessor(Node):
    """Prepare kaggle dataset for training

    * normalize and reshape the features
    * one-hot encode the labels
    """
    # dependencies and parameters
    data: pathlib.Path = dvc.deps(pathlib.Path("dataset"))
    dataset = zn.params("sign_mnist_train")
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


