import os
import sys
from datetime import datetime
from Insurance.exception import InsuranceException
from Insurance.logger import logging

FILE_NAME = "insurance.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"


class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifacts_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise InsuranceException(e, sys)

# Data Ingestion
class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.database_name = "INSURANCE"
            self.collection_name = "INSURANCE_COLLECTION"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifacts_dir, "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "features", FILE_NAME)
            self.train_data_file_path = os.path.join(self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME)
            self.test_data_file_path = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception as e:
            raise InsuranceException(e, sys)

    def to_dict(self) -> dict:
        try:
            return self.__dict__
        except Exception as e:
            InsuranceException(e, sys)

# Data Validation
class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifacts_dir, "data_validation")
            self.validation_report_file = os.path.join(self.data_validation_dir, "Data_Validation_Report.yaml")
            self.threshold = 0.2
            self.base_file_path = "insurance.csv"
        except Exception as e:
            raise InsuranceException(e, sys)
        
# Data Transformation
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_transformation_dir = os.path.join(training_pipeline_config.artifacts_dir, "data_transformation")
            self.transform_object_path = os.path.join(self.data_transformation_dir, "transformer", TRANSFORMER_OBJECT_FILE_NAME)
            self.transform_train_path = os.path.join(self.data_transformation_dir, "transformed", TRAIN_FILE_NAME.replace("csv", "npz"))
            self.transform_test_path = os.path.join(self.data_transformation_dir, "transformed", TEST_FILE_NAME.replace("csv", "npz"))
            self.target_encoder_path = os.path.join(self.data_transformation_dir, "target_encoder", TARGET_ENCODER_OBJECT_FILE_NAME)

        except Exception as e:
            raise InsuranceException(e, sys)
        
# Model Trainer
class ModelTrainingConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.model_dir = os.path.join(training_pipeline_config.artifacts_dir, "model_trainer")
            self.model_path = os.path.join(self.model_dir, "model", MODEL_FILE_NAME)
            self.excepted_accuracy = 0.75
            self.overfitting_threshold = 0.3
            
        except Exception as e:
            raise InsuranceException(e, sys)
        
# Model Evaluation
class ModelEvalutationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.change_threshold = 0.01


# Model Pusher
class ModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Creating a folder to save new model
        self.model_pusher_dir = os.path.join(training_pipeline_config.artifacts_dir, "model_pusher")
        self.saved_model_dir = os.path.join("saved_models")
        self.pusher_model_dir = os.path.join(self.model_pusher_dir, "saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_dir, MODEL_FILE_NAME)
        self.pusher_transformer_path = os.path.join(self.pusher_model_dir, TRANSFORMER_OBJECT_FILE_NAME)
        self.pusher_target_encoder_path = os.path.join(self.pusher_model_dir, TARGET_ENCODER_OBJECT_FILE_NAME)
