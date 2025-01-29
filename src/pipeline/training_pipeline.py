import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

logging.info("Training pipeline started.")
class TrainingPipeline:
    def __init__(self):
        pass

    def start_data_ingestion(self):
        try:
            # Initiate data ingestion
            data_ingestion = DataIngestion()
            train, test = data_ingestion.initiate_data_ingestion()
            return train, test
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, train_path, test_path):
        try:
            # Initiate data transformation
            transformer = DataTransformation()
            train_transformed, test_transformed, _ = transformer.initiate_data_transformation(train_path, test_path)
            return test_transformed, test_transformed
        except Exception as e:
            raise CustomException(e, sys)

    def model_trainer(self, train_transformed, test_transformed):
        try:
            # Initiate model training
            model_trainer = ModelTrainer()
            result, model = model_trainer.initiate_model_trainer(train_transformed, test_transformed)
            return result, model
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_training(self):
        try:
            train_path, test_path = self.start_data_ingestion()
            train_arr, test_arr = self.start_data_transformation(train_path, test_path)
            score, model = self.model_trainer(train_arr, test_arr)
            return score, model
        except Exception as e:
            raise CustomException(e, sys)
        
# if __name__ == "__main__":
#     trainer = TrainingPipeline()
#     score, model = trainer.start_training()
#     print(score, model)