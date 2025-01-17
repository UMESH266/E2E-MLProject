from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Initiate data ingestion
data_ingestion = DataIngestion()
train, test = data_ingestion.initiate_data_ingestion()

# Initiate data transformation
transformer = DataTransformation()
train_transformed, test_transformed, _ = transformer.initiate_data_transformation(train, test)

# Initiate model training
model_trainer = ModelTrainer()
result, model = model_trainer.initiate_model_trainer(train_transformed, test_transformed)
print(f"Best model {model} performance: ", result)