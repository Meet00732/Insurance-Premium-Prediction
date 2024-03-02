import os
import sys
import numpy as np
import pandas as pd
from Insurance.entity import config_entity
from Insurance.entity import artifact_entity
from Insurance.exception import InsuranceException
from Insurance.utils import get_collection_dataframe
import logging
from sklearn.model_selection import train_test_split

class DataIngestion:
    # Dividing data into train, test, validation
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise InsuranceException(e, sys)
        

    # Initiate data ingestion
    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:

        try:
            # Reading dataset from database
            logging.info(f"Reading Data as DataFrame")
            df:pd.DataFrame = get_collection_dataframe(self.data_ingestion_config.database_name,
                                                            self.data_ingestion_config.collection_name)
            
            logging.info(f"Save Data in Feature Store")

            # Replace na with NAN
            df.replace(to_replace='na', value=np.NAN, inplace=True)
            logging.info("Replaced na with NAN")

            # Save data in feature_store
            logging.info("Creating Feature store if not exist!")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            logging.info("Save entire data to Feature Store Folder!")
            df.to_csv(path_or_buf = self.data_ingestion_config.feature_store_file_path, header=True, index=False)

            # Spliting data
            logging.info("Spliting Dataset into train, test and validation")
            train_data, test_data = train_test_split(df, test_size = self.data_ingestion_config.test_size, random_state=101)

            # Create directory for dataset
            logging.info("Creating Dataset direction if not exist!")
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_data_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            # Saving train and test data to feature store
            logging.info("Save train and test data to feature store folder")
            train_data.to_csv(path_or_buf = self.data_ingestion_config.train_data_file_path, header=True, index=False)
            test_data.to_csv(path_or_buf = self.data_ingestion_config.test_data_file_path, header=True, index=False)

            # Prepare artifact folder
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path = self.data_ingestion_config.feature_store_file_path,
                train_data_file_path = self.data_ingestion_config.train_data_file_path,
                test_data_file_path = self.data_ingestion_config.test_data_file_path
            )

        except Exception as e:
            raise InsuranceException(e, sys)
        
