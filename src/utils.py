from ast import mod
import os
import sys

import dill
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, obj):
    """This function saves pickle file to artifacts folder by creating folder if doesn't exists."""
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    """This function evaluates models return model and the r2_score of the model."""
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i] # Extracting model from models dictionary

            # Fit model and train model
            model.fit(X_train, y_train)

            # Predictions on test data
            y_predictions = model.predict(X_test)

            # Evaluatoin of r2_score
            score = r2_score(y_test, y_predictions)

            report[list(models.keys())[i]] = score

            return report

    except Exception as e:
        raise CustomException(e, sys)
