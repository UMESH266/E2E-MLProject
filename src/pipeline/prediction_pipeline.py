import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


logging.info("Prediction pipeline initiated.")

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, new_data):
        try:
            # Load model and preprocessor
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_transform = preprocessor.transform(new_data)
            prediction = model.predict(data_transform)

            return prediction
        except Exception as e:
            raise CustomException(e, sys)