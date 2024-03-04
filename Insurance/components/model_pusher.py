import sys
import os
import numpy as np
import pandas as pd
from Insurance.entity import artifact_entity, config_entity
from Insurance.exception import InsuranceException
from Insurance.predictor import ModelRegistry
from Insurance import utils
from Insurance.logger import logging
from sklearn.metrics import r2_score
from Insurance.config import TARGET_COLUMN
from Insurance.entity.artifact_entity import DataTransformationArtifact, ModelTrainingArtifact, ModelPusherArtifact
from Insurance.entity.config_entity import ModelPusherConfig
from Insurance.predictor import ModelRegistry

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainingArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_registry = ModelRegistry(model_registry=self.model_pusher_config.saved_model_dir)

        except Exception as e:
            raise InsuranceException(e, sys)
    
    def initiate_model_pusher(self)->ModelPusherArtifact:
        try:

            # Loading Model, Transformer and Target Encoder
            logging.info(f"Loading Transformer, Model and Target Encoder")
            transformer = utils.load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = utils.load_object(file_path=self.model_trainer_artifact.model_path)
            target_encoder = utils.load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            # Model Pusher Directory
            logging.info(f"Saving Transformer, Model and Target Encoder")
            utils.save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            utils.save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            utils.save_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=target_encoder)

            
            # Saving final model after model evaluation
            logging.info(f"Saving Final Model after Model Evaluation")
            transformer_path = self.model_registry.get_latest_saved_transformed_path()
            model_path = self.model_registry.get_latest_saved_model_path()
            target_encoder_path = self.model_registry.get_latest_saved_target_encoder_path()


            utils.save_object(file_path=transformer_path, obj=transformer)
            utils.save_object(file_path=model_path, obj=model)
            utils.save_object(file_path=target_encoder_path, obj=target_encoder)


            logging.info(f"Creating Model Pusher Artifact")
            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                                                        saved_model_dir=self.model_pusher_config.saved_model_dir)
            
            return model_pusher_artifact

        except Exception as e:
            raise InsuranceException(e, sys)