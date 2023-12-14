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
    def prepare_base_model(model, input_shape, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        model.summary()

        return model

    def updated_base_model(self):
        self.model = self.prepare_base_model(
            model=self.model,
            input_shape=self.input_shape,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.learning_rate,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        logger.info("Saving model")
