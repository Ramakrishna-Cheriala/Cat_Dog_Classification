from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: Path
    local_dir_files: Path
    unzip_dirs: Path
    split_data_dir: Path
    original_data_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    learning_rate: float
    classes: int
    input_shape: list
    include_top: bool
    weights: str


@dataclass(frozen=True)
class TrainingModelConfig:
    root_dir: Path
    updated_base_model_path: Path
    trained_model_path: Path
    training_data: Path
    validation_data: Path
    epochs: int
    batch_size: int
    weights: str
    include_top: bool
    input_shape: list
    class_weights: dict
