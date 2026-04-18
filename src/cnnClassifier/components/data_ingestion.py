import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self) -> str:
        """
        Downloads file from Google Drive
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)

            if not os.path.exists(zip_download_dir):
                logger.info(f"Downloading data from {dataset_url} into {zip_download_dir}")

                file_id = dataset_url.split("/")[-2]
                prefix = "https://drive.google.com/uc?/export=download&id="
                gdown.download(prefix + file_id, zip_download_dir, quiet=False)

                logger.info("Download completed")
            else:
                logger.info("File already exists. Skipping download.")

            return zip_download_dir

        except Exception as e:
            logger.error(f"Error in downloading file: {e}")
            raise e


    def extract_zip_file(self):
        """
        Extracts zip file into directory
        """
        try:
            unzip_path = self.config.unzip_dir
            zip_file_path = self.config.local_data_file

            os.makedirs(unzip_path, exist_ok=True)

            logger.info(f"Extracting {zip_file_path} into {unzip_path}")

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info("Extraction completed")

        except Exception as e:
            logger.error(f"Error in extracting zip file: {e}")
            raise e