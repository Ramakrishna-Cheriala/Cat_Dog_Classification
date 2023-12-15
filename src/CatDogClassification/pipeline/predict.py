import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from src.CatDogClassification import logger


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def preprocess_image(self, img_path):
        target_size = (150, 150)
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self):
        model = load_model(os.path.join("artifacts", "training_model", "model.h5"))
        img_path = self.filename
        preprocessed_image = self.preprocess_image(img_path)

        # prediction = model.predict(preprocessed_image)
        prediction_prob = model.predict(preprocessed_image)[0][0]

        # Set a threshold for classification
        threshold = 0.5
        if prediction_prob >= threshold:
            prediction_class = "Dog"
        else:
            prediction_class = "Cat"

        logger.info(
            f"Predicted class: {prediction_class}, Probability: {prediction_prob}"
        )
        return [{"image": prediction_class}]
