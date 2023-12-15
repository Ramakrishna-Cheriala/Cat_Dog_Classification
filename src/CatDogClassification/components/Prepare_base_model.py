import os
from src.CatDogClassification import logger
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from pathlib import Path
from src.CatDogClassification.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = models.Sequential()
        self.input_shape = self.config.input_shape

    @staticmethod
    def prepare_base_model(model, input_shape, weights, include_top, learning_rate):
        base_model = tf.keras.applications.VGG16(
            weights=weights, include_top=include_top, input_shape=input_shape
        )
        base_model.trainable = False

        model = models.Sequential(
            [
                base_model,
                layers.Flatten(),
                layers.Dense(512, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.9
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        model.summary()

        return model

    def updated_base_model(self):
        self.model = self.prepare_base_model(
            model=self.model,
            input_shape=self.input_shape,
            weights=self.config.weights,
            include_top=self.config.include_top,
            learning_rate=self.config.learning_rate,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        logger.info("Saving model")
