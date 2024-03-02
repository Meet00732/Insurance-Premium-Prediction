import os
import sys
from datetime import datetime
from Insurance.exception import InsuranceException
from Insurance.logger import logging

FILE_NAME = "insurance.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"


class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifacts_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise InsuranceException(e, sys)


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.database_name = "insurance"
            self.collection_name = "premium"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifacts_dir, "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "features", FILE_NAME)
            self.train_data_file_path = os.path.join(self.data_ingestion_dir, "dataset", TRAIN_FILE_NAME)
            self.test_data_file_path = os.path.join(self.data_ingestion_dir, "dataset", TEST_FILE_NAME)
        except Exception as e:
            raise InsuranceException(e, sys)

    def to_dict(self) -> dict:
        try:
            return self.__dict__
        except Exception as e:
            InsuranceException(e, sys)
