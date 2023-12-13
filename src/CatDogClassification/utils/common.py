import os
import yaml
from box.exceptions import BoxValueError
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
import base64
from src.CatDogClassification import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path):
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded")
            return ConfigBox(content)
    except BoxValueError:
        return ValueError("yaml file is empty")
    except Exception as e:
        raise e


def create_directory(path_to_dir: list, verbose=True):
    for path in path_to_dir:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")
