import pathlib

import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from zntrack import Node, utils, zn
from zntrack.core import ZnTrackOption


class TFModel(ZnTrackOption):
    dvc_option = "outs"
    zn_type = utils.ZnTypes.RESULTS

    def get_filename(self, instance) -> pathlib.Path:
        """Filename depending on the instance node_name"""
        return pathlib.Path("nodes", instance.node_name, "model")

    def save(self, instance):
        """Serialize and save values to file"""
        model = self.__get__(instance, self.owner)
        file = self.get_filename(instance)
        model.save(file)

    def get_data_from_files(self, instance):
        """Load values from file and deserialize"""
        file = self.get_filename(instance)
        model = keras.models.load_model(file)
        return model


class MLModel(Node):
    # dependencies
    train_data = zn.deps()
    # outputs
    training_history = zn.plots()
    metrics = zn.metrics()
    # custom model output
    model = TFModel()
    # parameter
    epochs = zn.params()
    filters = zn.params([4])
    dense = zn.params([4])
    optimizer = zn.params("adam")

    def run(self):
        """Primary Node Method"""
        self.build_model()
        self.train_model()

    def train_model(self):
        """Train the model"""
        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        print(self.model.summary())

        history = self.model.fit(
            self.train_data.features,
            self.train_data.labels,
            validation_split=0.3,
            epochs=self.epochs,
            batch_size=64,
        )
        self.training_history = pd.DataFrame(history.history)
        self.training_history.index.name = "epoch"
        # use the last values for model metrics
        self.metrics = dict(self.training_history.iloc[-1])

    def build_model(self):
        """Build the model using keras.Sequential API"""

        inputs = keras.Input(shape=(28, 28, 1))
        cargo = inputs
        for filters in self.filters:
            cargo = layers.Conv2D(
                filters=filters, kernel_size=(3, 3), padding="same", activation="relu"
            )(cargo)
            cargo = layers.MaxPooling2D((2, 2))(cargo)

        cargo = layers.Flatten()(cargo)

        for dense in self.dense:
            cargo = layers.Dense(dense, activation="relu")(cargo)

        output = layers.Dense(25, activation="softmax")(cargo)

        self.model = keras.Model(inputs=inputs, outputs=output)
