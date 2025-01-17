import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV

# Create Model trainer config
@dataclass
class ModelTrainerConfig:
    model_trainer_file_path : str = os.path.join('artifacts', 'model.pkl')

# Create model trainer class
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # Method to train different models
    def initiate_model_trainer(self, train_array, test_array):
        """
        This function returns best performing model on the given dataset.
        """
        logging.info("Model training started.")
        try:
            logging.info("Spliting of train and test data")
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Support Vector Regressor": SVR()
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # Best model score
            best_model_score = max(model_report.values())

            # Best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # Best model 
            best_model = models[best_model_name]

            if best_model_score < 0.60:
                raise CustomException("No best model found.")
            
            logging.info("Best model found on test set.")
            save_object(self.model_trainer_config.model_trainer_file_path, obj=best_model)

            # Best model performance
            predicted = best_model.predict(X_test)
            best_score = r2_score(y_test, predicted)

            return best_score, best_model

        except Exception as e:
            raise CustomException(e, sys)

