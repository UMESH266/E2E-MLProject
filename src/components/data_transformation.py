import os
import sys

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.utils import save_object

# Create data transformation config to store path for storing preprocessor pickle file.
class DataTrasformationConfig:
    preprocessor_obj_file_path : str = os.path.join('artifacts', 'preprocessor.pkl')

logging.info("Data transformaiton config created.")

# Create data transformation classs
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTrasformationConfig()

    # Create data transformer object
    def get_data_transformer_object(self):
        """
        This function returns preprocessor object which imputes and transforms columns based on numerical or categorical feature.
        """
        try:
            logging.info("Dataset columns seggregation to numerical and categorical columns.")
            numercial_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            logging.info("Imputer and scaling steps pipeline creation.")
            numrical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy='median')),
                                                ('scalre', StandardScaler(with_mean=False))])
            
            categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                ('encoder', OneHotEncoder()),
                                                ("scaler", StandardScaler(with_mean=False))])

            logging.info("Preprocessor object creation transformer.")
            preprocessor = ColumnTransformer([
                ("numerical_pipeline", numrical_pipeline, numercial_columns),
                ("Categorical_pipeline", categorical_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    # Inititate data transformation
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function initiates data transformation and return transformed train and test data. Also saves preprocessor object as pickle file.
        """
        try:
            logging.info("Data transformation initiated.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading of train and test data completed.")

            preprocessor_object = self.get_data_transformer_object()
            logging.info("Preprocessor object collected.")

            # Feature selection
            target_column_name = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessor on training and testing dataframe.")
            input_feature_train_arr = preprocessor_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_object.transform(input_feature_test_df)

            # Concatination of transformed arr with target features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessor object
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor_object)

            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)