import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config


    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        logger.info("Validation generator created")


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        logger.info(f"Loading model from {path}")
        return tf.keras.models.load_model(path)


    def evaluation(self):
        try:
            self.model = self.load_model(self.config.path_of_model)
            self._valid_generator()

            logger.info("Starting model evaluation...")
            self.score = self.model.evaluate(self.valid_generator)

            logger.info(f"Evaluation results: Loss={self.score[0]}, Accuracy={self.score[1]}")

            self.save_score()

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise e


    def save_score(self):
        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1])
        }

        save_json(path=Path(self.config.metric_file_name), data=scores)
        logger.info("Scores saved successfully")


    def log_into_mlflow(self):
        try:
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                mlflow.log_params(self.config.all_params)

                mlflow.log_metrics({
                    "loss": self.score[0],
                    "accuracy": self.score[1]
                })

                if tracking_url_type_store != "file":
                    mlflow.keras.log_model(
                        self.model,
                        "model",
                        registered_model_name="VGG16Model"
                    )
                else:
                    mlflow.keras.log_model(self.model, "model")

            logger.info("MLflow logging completed")

        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            raise e