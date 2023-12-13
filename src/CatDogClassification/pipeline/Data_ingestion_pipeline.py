from src.CatDogClassification.config.configuration import ConfigurationManager
from src.CatDogClassification.entity.config_entity import DataIngestionConfig
from src.CatDogClassification.components.Data_ingestion import DataIngestion
from src.CatDogClassification import logger


STAGE_NAME = "Data Ingestion"


class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        # data_ingestion.dowenload_file()
        data_ingestion.extract_zip_file()
        data_ingestion.data_splitting()


if __name__ == "__main__":
    try:
        logger.info(f"{STAGE_NAME} started")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f"{STAGE_NAME} finished")
    except Exception as e:
        logger.exception(e)
        raise e
