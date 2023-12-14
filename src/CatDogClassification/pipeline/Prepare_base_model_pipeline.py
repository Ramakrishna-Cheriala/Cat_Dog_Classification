from src.CatDogClassification.config.configuration import ConfigurationManager
from src.CatDogClassification.entity.config_entity import PrepareBaseModelConfig
from src.CatDogClassification.components.Prepare_base_model import PrepareBaseModel
from src.CatDogClassification import logger


STAGE_NAME = "Prepare Base Model"


class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepared_base_model()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.updated_base_model()


if __name__ == "__main__":
    try:
        logger.info(f"{STAGE_NAME} started")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f"{STAGE_NAME} finished")
    except Exception as e:
        logger.exception(e)
        raise e
