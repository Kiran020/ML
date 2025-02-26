from src.constants import *
from src.config.configuration import *
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = TRAIN_FILE_PATH
    test_data_path: str = TEST_FILE_PATH
    raw_data_path: str = RAW_FILE_PATH

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting Data Ingestion...")

            # Ensure the dataset exists
            df = pd.read_csv('D:/ML_PROJECT/ML/Jupiter/finalTrain.csv')
            logging.info(f"Loaded dataset with shape: {df.shape}")

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved successfully.")

            # Splitting the dataset
            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)
            logging.info(f"Train shape: {train_set.shape}, Test shape: {test_set.shape}")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path), exist_ok=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully.")
            return self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path

        except Exception as e:
            logging.error(f"Error in Data Ingestion: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    # train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_training(train_arr,test_arr))
    logging.info("Data Transformation completed.")
