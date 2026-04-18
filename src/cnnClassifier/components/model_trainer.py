import os
import tensorflow as tf
from pathlib import Path

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config


    def get_base_model(self):
        logger.info(f"Loading base model from {self.config.updated_base_model_path}")
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        logger.info("Base model loaded successfully")


    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
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

        if self.config.params_is_augmentation:
            logger.info("Using data augmentation")

            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            logger.info("No data augmentation used")
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        logger.info("Train & validation generators created")


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)


    def train(self):
        try:
            self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
            self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

            logger.info("Starting training...")

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.config.trained_model_path,
                    save_best_only=True
                )
            ]

            self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator,
                callbacks=callbacks
            )

            logger.info("Training completed")

            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )

            logger.info(f"Model saved at {self.config.trained_model_path}")

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise e