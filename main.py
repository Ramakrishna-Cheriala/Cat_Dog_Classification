from src.CatDogClassification import logger
from src.CatDogClassification.pipeline.Data_ingestion_pipeline import (
    DataIngestionPipeline,
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
