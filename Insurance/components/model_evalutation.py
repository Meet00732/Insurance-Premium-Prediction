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
            self.model_registry = ModelRegistry()
        except Exception as e:
            raise InsuranceException(e, sys)
    

    def initaite_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            latest_dir_path = self.model_registry.get_latest_dir_path()

            if latest_dir_path == None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, improved_score=None)
                logging.info(f"Model Evaluation Artifact: {model_eval_artifact}")
            
                return model_eval_artifact
            
            # Find location of previous model
            logging.info("Finding location (path) of previous model")
            transformer_path = self.model_registry.get_latest_transformer_path()
            model_path = self.model_registry.get_latest_model_path()
            target_encoder_path = self.model_registry.get_latest_target_encoder_path()

            
            # Loading data of previous model
            logging.info(f"Loading data that was find.")
            prev_transformer = utils.load_object(file_path=transformer_path)
            prev_model = utils.load_object(file_path=model_path)
            prev_target_encoder = utils.load_object(file_path=target_encoder_path)


            # Loading Current Model
            logging.info(f"Loading current model")
            curr_transformer = utils.load_object(file_path=self.data_transformation_artifact.transform_object_path)
            curr_model = utils.load_object(file_path=self.model_trainer_artifact.model_path)
            curr_target_encoder = utils.load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            
            # Reading test data
            logging.info(f"Reading the test data!")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_file_path)
            y_true = test_df[TARGET_COLUMN]

            # Encoding
            logging.info(f"Performing Feature Encoding on test data for prev model")
            prev_X_test_names = list(prev_transformer.feature_names_in_)
            for name in prev_X_test_names:
                if test_df[name].dtype == 'O':
                    test_df[name] = prev_target_encoder.fit_transform(test_df[name])
            
            X_test_arr = prev_transformer.transform(test_df[prev_X_test_names])
            prev_y_pred = prev_model.predict(X_test_arr)


            # Comparing the model
            logging.info(f"Comparing the prev model and curr model")
            prev_model_score = r2_score(y_true=y_true, y_pred=prev_y_pred)
            

            # Accuracy of current model
            curr_X_test_name = list(curr_transformer.feature_names_in_)
            curr_X_test_arr = curr_transformer.transform(test_df[curr_X_test_name])

            curr_y_pred = curr_model.predict(curr_X_test_arr)
            curr_model_score = r2_score(y_true=y_true, y_pred=curr_y_pred)

            # Comaring both model
            logging.info(f"Comparing r2 score for both models")
            if curr_model_score < prev_model_score:
                logging.info(f"Current Trained Model is not better than previous one")
                raise Exception(f"Current Trained Model is not better than previous one")
            

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, 
                                                                          improved_score=curr_model_score - prev_model_score)
            
            return model_eval_artifact

        except Exception as e:
            raise InsuranceException(e, sys)