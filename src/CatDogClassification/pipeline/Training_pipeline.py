from src.CatDogClassification.config.configuration import ConfigurationManager
from src.CatDogClassification.entity.config_entity import TrainingModelConfig
from src.CatDogClassification.components.Training_model import Training
from src.CatDogClassification import logger


class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_model()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()


STAGE_NAME = "Training model"
if __name__ == "__main__":
    try:
        logger.info(f"{STAGE_NAME} started")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f"{STAGE_NAME} finished")
    except Exception as e:
        logger.exception(e)
        raise e
