import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.CatDogClassification.entity.config_entity import (
    TrainingModelConfig,
)
from src.CatDogClassification import logger
from pathlib import Path
from src.CatDogClassification.utils.common import read_yaml, create_directory, save_json


class Training:
    def __init__(self, config: TrainingModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
            # os.path.join("artifacts","prepare_base_model","base_model_updated")
        )

    def train_valid_generator(self):
        # train_data = ImageDataGenerator(rescale = 1./255)
        train_data = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        test_data = ImageDataGenerator(rescale=1.0 / 255)
        train_generator = train_data.flow_from_directory(
            self.config.training_data,
            batch_size=20,
            target_size=(150, 150),
            class_mode="binary",
        )
        validation_generator = train_data.flow_from_directory(
            self.config.validation_data,
            batch_size=20,
            target_size=(150, 150),
            class_mode="binary",
        )

        self.model.fit(
            train_generator,
            epochs=self.config.epochs,
            validation_data=validation_generator,
            validation_steps=50,
            class_weight=self.config.class_weights,
        )

        scores = self.model.evaluate(
            validation_generator, batch_size=self.config.batch_size
        )
        # print("Validation Loss:", scores[0])
        # print("Validation Accuracy:", scores[1])
        score = {"loss": scores[0], "accuracy": scores[1]}
        save_json(path=Path("scores.json"), data=score)
        # logger.info("Model Scores saved in scores.json.......")

        self.save_model(path=self.config.trained_model_path, model=self.model)
        # logger.info("Model training and saving completed...........")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
