import sys
import os
import numpy as np
import pandas as pd
from Insurance.entity import artifact_entity, config_entity
from Insurance.exception import InsuranceException
from Insurance.predictor import SavedModels
from Insurance import utils
from Insurance.logger import logging


class ModelEvaluation:
    def __init__(self, model_evaluation_config: config_entity.ModelEvalutationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainingArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.saved_models = SavedModels()
        except Exception as e:
            raise InsuranceException(e, sys)
    

    def initaite_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            latest_dir_path = self.saved_models.get_latest_dir_path()

            if latest_dir_path == None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, improved_accuracy=None)
                logging.info(f"Model Evaluation Artifact: {model_eval_artifact}")
                return model_eval_artifact
        except Exception as e:
            raise InsuranceException(e, sys)