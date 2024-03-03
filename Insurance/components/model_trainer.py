import sys
import os
import numpy as np
import pandas as pd
from Insurance.entity import artifact_entity, config_entity
from Insurance.exception import InsuranceException
from Insurance import utils
from Insurance.logger import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Define Model and then Train it.
# If running model accuracy = 80 and when we train our model with new data and get accuracy lesser than that 
# then we use the current model. set (threshold) for this.
# Check for overfitting and underfitting

class ModelTrainer:
    def __init__(self, model_trainer_config: config_entity.ModelTrainingConfig,
                 transformed_data_artifact: artifact_entity.DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.transformed_data_artifact = transformed_data_artifact
        except Exception as e:
            raise InsuranceException(e, sys)
        
    
    def train_model(self, X, y):
        try:
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            return lr_model

        except Exception as e:
            raise InsuranceException(e, sys)
        


    def initiate_model_trainer(self) -> artifact_entity.ModelTrainingArtifact:
        try:
            logging.info(f"Reading transformed data in numpy array form")
            train_arr = utils.load_numpy_data(self.transformed_data_artifact.transform_train_path)
            test_arr = utils.load_numpy_data(self.transformed_data_artifact.transform_test_path)

            logging.info(f"Spliting data X_train and y_train")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]


            logging.info(f"Building a model")
            model = self.train_model(X_train, y_train)

            logging.info(f"Predicting Model for both train and test")
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            logging.info(f"Calculating r2 score for Linear regression model")
            r2_score_train = r2_score(y_train, y_pred_train)
            r2_score_test = r2_score(y_test, y_pred_test)

            logging.info(f"Checking Model performance if exceeds the threshold")
            if r2_score_test < self.model_trainer_config.excepted_accuracy:
                raise Exception(f"Model is not performing good \
                                Expected Accuracy: {self.model_trainer_config.excepted_accuracy} and Model r2 score: {r2_score_test}")

            logging.info(f"Checking for Overfitting")
            diff = np.abs(r2_score_test - r2_score_train)
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Model is exceeding overfitting threshold \
                                Overfitting Threshold: {self.model_trainer_config.overfitting_threshold} \
                                Model score difference: {diff}")
            
            utils.save_object(self.model_trainer_config.model_path, obj=model)

            model_trainer_artifact = artifact_entity.ModelTrainingArtifact(model_path=self.model_trainer_config.model_path,
                                                                           r2_train_score = r2_score_train,
                                                                           r2_test_score = r2_score_test)
            
            return model_trainer_artifact

        except Exception as e:
            raise InsuranceException(e, sys)