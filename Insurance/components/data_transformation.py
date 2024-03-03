import sys
import os
import numpy as np
import pandas as pd
from Insurance.entity import artifact_entity, config_entity
from Insurance.exception import InsuranceException
from Insurance import utils
from Insurance.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from Insurance.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder


# Handling Missing values (imputation)
# Outliers handling
# Handling unbalanced data
# Feature Encoding


class DataTransformation:

    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise InsuranceException(e, sys)
        
    # Function that return pipeline as in sklearn
    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            # Hanldes missing values
            logging.info(f"Handling Missing Values")
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)

            # Handles outliers
            logging.info(f"Handling Outliers")
            robust_scaler = RobustScaler()

            # Pipeline
            logging.info(f"Creating Pipeline")
            pipeline = Pipeline(steps=[
                ('SimpleImputer', simple_imputer),
                ('RobustScaler', robust_scaler)
            ])

            return pipeline
        
        except Exception as e:
            raise InsuranceException(e, sys)


    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
            logging.info(f"Reading Train and Test data in transformation")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_data_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_file_path)

            logging.info(f"Separating features and target from train and test data")
            X_train = train_df.drop(TARGET_COLUMN, axis=1)
            X_test = test_df.drop(TARGET_COLUMN, axis=1)

            y_train = train_df[TARGET_COLUMN]
            y_test = test_df[TARGET_COLUMN]

            logging.info("Performing Label Encoding")
            label_encoder = LabelEncoder()

            y_train_arr = y_train.squeeze()
            y_test_arr = y_test.squeeze()

            logging.info("Performing Label Encoding on X_train and X_test")
            for col in X_train.columns:
                if X_train[col].dtype == 'O':
                    X_train[col] = label_encoder.fit_transform(X_train[col])
                    X_test[col] = label_encoder.fit_transform(X_test[col])
                else:
                    X_train[col] = X_train[col]
                    X_test[col] = X_test[col]

            
            # Calling pipeline
            logging.info(f"Creating data transformation pipeline")
            transformation_pipeline = DataTransformation.get_data_transformer_object()
            logging.info(f"Performing .fit on X_train")
            transformation_pipeline.fit(X_train)

            logging.info(f"Performing .transform on X_train and X_test")
            X_train_arr = transformation_pipeline.transform(X_train)
            X_test_arr = transformation_pipeline.transform(X_test)

            logging.info(f"Converting to numpy")
            train_arr = np.c_[X_train_arr, y_train_arr]
            test_arr = np.c_[X_test_arr, y_test_arr]

            utils.save_to_numpy(file_path=self.data_transformation_config.transform_train_path, array=train_arr)
            utils.save_to_numpy(file_path=self.data_transformation_config.transform_test_path, array=test_arr)

            logging.info(f"Saving Tranformer Pipeline")
            utils.save_object(file_path=self.data_transformation_config.transform_object_path, obj=transformation_pipeline)

            logging.info(f"Saving Encoder")
            utils.save_object(file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transform_train_path=self.data_transformation_config.transform_train_path,
                transform_test_path=self.data_transformation_config.transform_test_path,
                transform_encoder_path = self.data_transformation_config.target_encoder_path
            )

            return data_transformation_artifact


        except Exception as e:
            raise InsuranceException(e, sys)