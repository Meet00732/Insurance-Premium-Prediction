import sys
import os
import numpy as np
import pandas as pd
from Insurance.entity import artifact_entity, config_entity
from Insurance.exception import InsuranceException
from Insurance import utils
from Insurance.logger import logging
from Insurance.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from Insurance.entity.config_entity import ModelTrainingConfig, ModelEvalutationConfig, ModelPusherConfig
from Insurance.entity import config_entity
from Insurance.components.data_ingestion import DataIngestion
from Insurance.components.data_validation import DataValidation
from Insurance.components.data_transformation import DataTransformation
from Insurance.components.model_trainer import ModelTrainer
from Insurance.components.model_evalutation import ModelEvaluation
from Insurance.components.model_pusher import ModelPusher


def training_pipeline():
    try:
        # 1. Data Ingestion
        logging.info(f"Step - 1: Data Ingestion")
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        logging.info(f"{data_ingestion_config.to_dict()}")

        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # 2. Data Validation
        logging.info(f"Step - 2: Data Validation")
        data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config,
                                         data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initate_data_validation()


        # 3. Data Transformation
        logging.info(f"Step - 3: Data Transformation")
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                 data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        
        # 4. Model Trainer
        logging.info(f"Step - 4: Model Training")
        model_trainer_config = config_entity.ModelTrainingConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                     transformed_data_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        # 5. Model Evaluation
        logging.info(f"Step - 5: Model Evaluation")
        model_evaluation_config = config_entity.ModelEvalutationConfig(training_pipeline_config=training_pipeline_config)
        model_evaluation = ModelEvaluation(model_evaluation_config=model_evaluation_config,
                                           data_ingestion_artifact=data_ingestion_artifact,
                                           data_transformation_artifact=data_transformation_artifact,
                                           model_trainer_artifact=model_trainer_artifact)
        model_evaluation.initaite_model_evaluation()

        # 6. Model Pusher
        logging.info(f"Step - 6: Model Pusher")
        model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config=training_pipeline_config)
        model_pusher = ModelPusher(model_pusher_config=model_pusher_config,
                                   data_transformation_artifact=data_transformation_artifact,
                                   model_trainer_artifact=model_trainer_artifact)
        model_pusher_artifact = model_pusher.initiate_model_pusher()
        
        logging.info(f"Training Pipeline Completed")

    except Exception as e:
        raise InsuranceException(e, sys)