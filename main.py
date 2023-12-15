from src.CatDogClassification import logger
from src.CatDogClassification.pipeline.Data_ingestion_pipeline import (
    DataIngestionPipeline,
)
from src.CatDogClassification.pipeline.Prepare_base_model_pipeline import (
    PrepareBaseModelPipeline,
)
from src.CatDogClassification.pipeline.Training_pipeline import (
    TrainingPipeline,
)


STAGE_NAME = "Data Ingestion"

try:
    logger.info(f"{STAGE_NAME} started")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f"{STAGE_NAME} finished")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f"{STAGE_NAME} started")
    obj = PrepareBaseModelPipeline()
    obj.main()
    logger.info(f"{STAGE_NAME} finished")
except Exception as e:
    logger.exception(e)
    raise e


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
