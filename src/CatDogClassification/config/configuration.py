from src.CatDogClassification.constants import *
from src.CatDogClassification.utils.common import read_yaml, create_directory
from src.CatDogClassification.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingModelConfig,
)


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

    def get_prepared_base_model(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directory([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=config.root_dir,
            base_model_path=config.base_model_path,
            updated_base_model_path=config.updated_base_model_path,
            learning_rate=self.params.LEARNING_RATE,
            classes=self.params.CLASSES,
            input_shape=self.params.INPUT_SHAPE,
            include_top=self.params.INCLUDE_TOP,
            weights=self.params.WEIGHTS,
        )

        return prepare_base_model_config

    def get_training_model(self) -> TrainingModelConfig:
        training = self.config.training_model
        prepare_base_model = self.config.prepare_base_model
        training_data = os.path.join(self.config.data_ingestion.split_data_dir, "train")
        validation_data = os.path.join(
            self.config.data_ingestion.split_data_dir, "validation"
        )
        create_directory([training.root_dir])

        prepare_training_model = TrainingModelConfig(
            root_dir=training.root_dir,
            updated_base_model_path=prepare_base_model.updated_base_model_path,
            trained_model_path=training.trained_model_path,
            training_data=Path(training_data),
            validation_data=Path(validation_data),
            epochs=self.params.EPOCHS,
            batch_size=self.params.BATCH_SIZE,
            weights=self.params.WEIGHTS,
            include_top=self.params.INCLUDE_TOP,
            input_shape=self.params.INPUT_SHAPE,
            class_weights=self.params.CLASS_WEIGHTS,
        )

        return prepare_training_model
