from src.CatDogClassification.constants import *
from src.CatDogClassification.utils.common import read_yaml, create_directory
from src.CatDogClassification.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    def __init__(
        self, config_filepath=Config_File_Path, param_filepath=Params_File_Path
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(param_filepath)

        create_directory([self.config.main_dir])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directory([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_dir_files=config.local_dir_files,
            unzip_dirs=config.unzip_dirs,
            split_data_dir=config.split_data_dir,
            original_data_dir=config.original_data_dir,
        )

        return data_ingestion_config
