import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from zntrack import Node, zn


class EvaluateModel(Node):
    # dependencies
    ml_model = zn.deps()
    test_data = zn.deps()
    # metrics
    metrics = zn.metrics()
    confusion_matrix = zn.plots(template="confusion", x="predicted", y="actual")

    def run(self):
        """Primary Node Method"""
        loss, accuracy = self.ml_model.evaluate(
            self.test_data.features, self.test_data.labels
        )
        self.metrics = {"loss": loss, "accuracy": accuracy}

        prediction = self.ml_model.predict(self.test_data.features)

        self.confusion_matrix = pd.DataFrame(
            [
                {"actual": np.argmax(true), "predicted": np.argmax(false)}
                for true, false in zip(self.test_data.labels, prediction)
            ]
        )

    def plot_confusion_matrix(self):
        cf_mat = confusion_matrix(
            self.confusion_matrix["actual"], self.confusion_matrix["predicted"]
        )
        sns.heatmap(cf_mat)
