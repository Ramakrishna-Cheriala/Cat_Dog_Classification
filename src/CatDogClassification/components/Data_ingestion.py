import os
import urllib.request as request
import zipfile
from src.CatDogClassification import logger
import shutil
import gdown
import requests
from pathlib import Path
from src.CatDogClassification.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def dowenload_file(self):
        # create_directory([self.config.local_dir_files])
        if not os.path.exists(self.config.local_dir_files):
            filename = gdown.download(
                self.config.source_url, self.config.local_dir_files, quiet=False
            )
            logger.info(f"\n{filename} dowenloaded\n")
        else:
            logger.info(f"\nfile already exists")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dirs
        print(self.config.local_dir_files)
        os.makedirs(unzip_path, exist_ok=True)
        logger.info("\nExtracting zip file")
        # filepath = r"C:\Users\ramak\OneDrive\Desktop\P2\CatDogClassification\Data\main_data.zip"
        filepath = Path("Data/main_data.zip")
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

    def data_splitting(self):  # Dividing the data into train, test, and validation sets
        if not os.path.exists(self.config.split_data_dir):
            logger.info("\nCreating train, test and validation sets")
            os.makedirs(self.config.split_data_dir)
            train_dir = os.path.join(self.config.split_data_dir, "train")
            validation_dir = os.path.join(self.config.split_data_dir, "validation")
            test_dir = os.path.join(self.config.split_data_dir, "test")

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(validation_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            train_cats_dir = os.path.join(train_dir, "cat")
            train_dogs_dir = os.path.join(train_dir, "dog")
            test_cats_dir = os.path.join(test_dir, "cat")
            test_dogs_dir = os.path.join(test_dir, "dog")
            validation_cats_dir = os.path.join(validation_dir, "cat")
            validation_dogs_dir = os.path.join(validation_dir, "dog")

            os.makedirs(train_cats_dir, exist_ok=True)
            os.makedirs(train_dogs_dir, exist_ok=True)
            os.makedirs(test_cats_dir, exist_ok=True)
            os.makedirs(test_dogs_dir, exist_ok=True)
            os.makedirs(validation_cats_dir, exist_ok=True)
            os.makedirs(validation_dogs_dir, exist_ok=True)

            # Copying the images to the train, test and validation directories

            # Copying cat and dogs images to train dir (1000 images (0 - 1000))
            logger.info("\nCopying images to train directory")
            cat_names = [f"cat.{i}.jpg" for i in range(1000)]
            for name in cat_names:
                # shutil.copy(os.path.join(self.config.unzip_dir, name), train_cats_dir)
                src = os.path.join(self.config.unzip_dirs, "main_data", name)
                destination = os.path.join(train_cats_dir, name)
                shutil.copy(src, destination)

            dog_names = [f"dog.{i}.jpg" for i in range(1000)]
            for name in dog_names:
                # shutil.copy(os.path.join(self.config.unzip_dir, name), train_dogs_dir)
                src = os.path.join(self.config.unzip_dirs, "main_data", name)
                destination = os.path.join(train_dogs_dir, name)
                shutil.copy(src, destination)

            # Copying cat and dog images to validation dir (500 images (1000 - 1500))
            logger.info("\nCopying images to test directory")
            cat_names = [f"cat.{i}.jpg" for i in range(1000, 1500)]
            for name in cat_names:
                # shutil.copy(os.path.join(self.config.unzip_dir, name), validation_cats_dir)
                src = os.path.join(self.config.unzip_dirs, "main_data", name)
                destination = os.path.join(validation_cats_dir, name)
                shutil.copy(src, destination)

            dog_names = [f"dog.{i}.jpg" for i in range(1000, 1500)]
            for name in dog_names:
                # shutil.copy(os.path.join(self.config.unzip_dir, name), validation_dogs_dir)
                src = os.path.join(self.config.unzip_dirs, "main_data", name)
                destination = os.path.join(validation_dogs_dir, name)
                shutil.copy(src, destination)

            # Copying cat and dog images to test dir(500 images (1500 - 2000))
            logger.info("\nCopying images to validation directory")
            cat_names = [f"cat.{i}.jpg" for i in range(1500, 2000)]
            for name in cat_names:
                # shutil.copy(os.path.join(self.config.unzip_dir, name), test_cats_dir)
                src = os.path.join(self.config.unzip_dirs, "main_data", name)
                destination = os.path.join(test_cats_dir, name)
                shutil.copy(src, destination)

            dog_names = [f"dog.{i}.jpg" for i in range(1500, 2000)]
            for name in dog_names:
                # shutil.copy(os.path.join(self.config.unzip_dir, name), test_dogs_dir)
                src = os.path.join(self.config.unzip_dirs, "main_data", name)
                destination = os.path.join(test_dogs_dir, name)
                shutil.copy(src, destination)

            logger.info("\nCopying of data is complete.")

        else:
            logger.info("\nFiles Already Exists")
