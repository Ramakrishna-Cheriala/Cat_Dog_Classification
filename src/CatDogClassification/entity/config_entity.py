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
