from Insurance.logger import logging
from Insurance.exception import InsuranceException
from Insurance.utils import get_collection_dataframe
from Insurance.entity.config_entity import DataIngestionConfig
from Insurance.entity import config_entity

import os
import sys

# def test_logger_exception():
#     try:
#         logging.info("Starting the test logger and exception!")
#         result = 3 / 0
#         print(result)
#         logging.info("Ending point of the test logger and exception!")
#     except Exception as e:
#         logging.debug(str(e))
#         raise InsuranceException(e, sys)
    

if __name__ == "__main__":
    try:
        # test_logger_exception()
        
        # get_collection_dataframe(databaseName="INSURANCE", collectionName="INSURANCE_COLLECTION")
        
        # Defining training pipeline
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config = training_pipeline_config)
        print(data_ingestion_config.to_dict())

    except Exception as e:
        InsuranceException(e, sys)