
import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras

from cnnClassifier import logger
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config):
        self.config = config


    def _valid_generator(self):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.30
        )

        self.valid_generator = datagen.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size
        )


    def evaluation(self):
        self.model = tf.keras.models.load_model(self.config.path_of_model)
        self._valid_generator()

        self.score = self.model.evaluate(self.valid_generator)

        self.save_score()


    def save_score(self):
        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1])
        }

        Path(self.config.metric_file_name).parent.mkdir(parents=True, exist_ok=True)

        save_json(self.config.metric_file_name, scores)


    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1]
            })

            mlflow.keras.log_model(self.model, "model")

