# Data type
# Unwanted data finding
# Data cleaning
# Make sure train, test and insurance.csv are same
import os
import sys
from Insurance.exception import InsuranceException
import pandas as pd
import numpy as np
from typing import Optional
from Insurance.entity import config_entity, artifact_entity
from Insurance.logger import logging
from scipy.stats import ks_2samp
from Insurance.config import TARGET_COLUMN
from Insurance import utils


class DataValidation:
    def __init__(self, data_validation_config: config_entity.DataValidationConfig,
                    data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            # logging.info(f"*****************Data Validation**********************")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error_report = dict()
        except Exception as e:
            raise InsuranceException(e, sys)


    def handling_missing_values(self, df:pd.DataFrame, error_key: str) -> Optional[pd.DataFrame]:
        try:
            threshold = self.data_validation_config.threshold
            missing_data = df.isna().sum() / df.shape[0]
            logging.info(f"Selecting columns which are above threshold value of {threshold}")
            drop_columns = missing_data[missing_data > threshold].index

            logging.info(f"Columns to drop: {list(drop_columns)}")
            self.validation_error_report[error_key] = list(drop_columns)
            df.drop(list(drop_columns), axis=1, inplace=True)

            if len(df.columns) == 0:
                return None
            return df
        
        except Exception as e:
            raise InsuranceException(e, sys)
        

    def is_required_feature_exists(self, original_data: pd.DataFrame, current_data: pd.DataFrame, error_key: str) -> bool:
        try:
            original_columns = original_data.columns
            current_columns = current_data.columns

            missing_cols = []
            for original_col in original_columns:
                if original_col not in current_columns:
                    missing_cols.append(original_col)
            
            if len(missing_cols) > 0:
                self.validation_error_report[error_key] = missing_cols
                return False
            return True

        except Exception as e:
            raise InsuranceException(e, sys)
        

    def data_drift(self, original_data: pd.DataFrame, current_data: pd.DataFrame, error_key: str):
        # Hypothesis Testing
        try:
            drift_report = dict()
            original_columns = original_data.columns
            current_columns = current_data.columns

            for original_col in original_columns:
                orig_data, curr_data = original_data[original_col], current_data[original_col]

                same_distribution = ks_2samp(orig_data, curr_data)

                if same_distribution.pvalue > 0.05: # p_value > 0.05
                    # Accept Null Hypothesis
                    drift_report[original_col] = {
                        'p-value': float(same_distribution.pvalue),
                        'same_distribution': True
                    }
                else:
                    drift_report[original_col] = {
                        'p-value': float(same_distribution.pvalue),
                        'same_distribution': False
                    }

            # Updating validation report
            self.validation_error_report[error_key] = drift_report

        except Exception as e:
            raise InsuranceException(e, sys)

    def initate_data_validation(self):
        # Read the data
        try:
            logging.info(f"Reading Original Dataframe")
            original_df = pd.read_csv(self.data_validation_config.base_file_path)

            logging.info(f"Replacing na to NaN values")
            original_df.replace({"na": np.NAN}, inplace=True)

            logging.info(f"Dropping null values with threshold check")
            self.handling_missing_values(original_df, "Missing_values_orignal_df")


            # Reading train data
            logging.info(f"Reading Training and Testing data")
            if os.path.exists(self.data_ingestion_artifact.train_data_file_path):
                logging.info("File exists, proceeding to read.")
            else:
                logging.info("File does not exist. Check the data ingestion process.")

            train_df = pd.read_csv(self.data_ingestion_artifact.train_data_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_data_file_path)

            # Drop missing values
            logging.info(f"Handling missing values for train and test data")
            train_df = self.handling_missing_values(train_df, "Missing_values_train_df")
            test_df = self.handling_missing_values(test_df, "Missing_values_test_df")

            # Updating datatype of columns
            logging.info(f"Converting Data to Float Type")
            exclude_col = [TARGET_COLUMN]
            original_df = utils.convert_to_float(original_df, exclude_col=exclude_col)
            train_df = utils.convert_to_float(train_df, exclude_col=exclude_col)
            test_df = utils.convert_to_float(test_df, exclude_col=exclude_col)

            # Check if all cols present or not
            logging.info(f"Checking if all required columns are present or not in train_df")
            train_df_status = self.is_required_feature_exists(original_data=original_df, current_data=train_df,
                                                              error_key="Missing_Columns_in_train_df")
            
            logging.info(f"Checking if all required columns are present or not in test_df")
            test_df_status = self.is_required_feature_exists(original_data=original_df, current_data=test_df,
                                                              error_key="Missing_Columns_in_test_df")
            

            if train_df_status:
                logging.info(f"All columns are present in both original and training data. Hence, checking data drift")
                self.data_drift(original_data=original_df, current_data=train_df,
                                error_key="Data_drift_in_train_df")
            
            if test_df_status:
                logging.info(f"All columns are present in both original and testing data. Hence, checking data drift")
                self.data_drift(original_data=original_df, current_data=test_df,
                                error_key="Data_drift_in_test_df")
                
            # Write report in yaml
            logging.info(f"Writing validation report in yaml")
            utils.write_report_yaml(self.data_validation_config.validation_report_file, self.validation_error_report)


            # Creating Data Validation Artifact
            logging.info(f"Initializing data validation artifact")
            data_validation_artifact = artifact_entity.DataValidationArtifact(self.data_validation_config.validation_report_file)

            return data_validation_artifact

        except Exception as e:
            raise InsuranceException(e, sys)