from Insurance.logger import logging
from Insurance.exception import InsuranceException
from Insurance.utils import get_collection_dataframe
from Insurance.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from Insurance.entity.config_entity import ModelTrainingConfig, ModelEvalutationConfig
from Insurance.entity import config_entity
from Insurance.components.data_ingestion import DataIngestion
from Insurance.components.data_validation import DataValidation
from Insurance.components.data_transformation import DataTransformation
from Insurance.components.model_trainer import ModelTrainer
from Insurance.components.model_evalutation import ModelEvaluation
from Insurance.entity import artifact_entity
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
        # print(data_ingestion_config.to_dict())

        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()


        # Data validation
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config,
                                         data_ingestion_artifact=data_ingestion_artifact)
        
        data_validation_artifact = data_validation.initate_data_validation()


        # Data Transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                 data_ingestion_artifact=data_ingestion_artifact)

        data_transformation_artifact = data_transformation.initiate_data_transformation()


        # Model Trainer
        model_trainer_config = ModelTrainingConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                     transformed_data_artifact=data_transformation_artifact)
        
        model_trainer_artifact = model_trainer.initiate_model_trainer()


        # Model Evaluation
        model_evaluation_config = ModelEvalutationConfig(training_pipeline_config=training_pipeline_config)
        model_evaluation = ModelEvaluation(model_evaluation_config=model_evaluation_config,
                                           data_ingestion_artifact=data_ingestion_artifact,
                                           data_transformation_artifact=data_transformation_artifact,
                                           model_trainer_artifact=model_trainer_artifact)
        
        model_evaluation_artifact = model_evaluation.initaite_model_evaluation()

    except Exception as e:
        InsuranceException(e, sys)