from math import log
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd

# Create dataclass to create variables for different paths to store data
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', "train.csv")
    test_data_path:str = os.path.join('artifacts', "test.csv")
    raw_data_path:str = os.path.join('artifacts', "data.csv")

# Create data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # Create data ingestion inititation method
    def initiate_data_ingestion(self):
        logging.info("Data ingestion initiated.")
        # Create exception handler
        try:
            logging.info("Data fetching initiated.")
            # Fetch data, split data, store data and return paths to train, test and raw data
            data = pd.read_csv('notebook\data\stud.csv')
            train, test = train_test_split(data, test_size=0.20, random_state=42)

            # Create or find artifact directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw, train, and test data
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data split as train and test set, saved to artifacts.")
            # Return file paths
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Error raised in data ingestion stage.")
            raise CustomException(e, sys)
    
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()