import numpy as np
import pandas as pd
import pymongo
import os
import sys
from Insurance.exception import InsuranceException
from Insurance.config import mongo_client
from Insurance.logger import logging
import yaml
import dill

def get_collection_dataframe(databaseName: str, collectionName: str) -> pd.DataFrame:
    try:
        logging.info(f"Reading Data from Database: {databaseName} and Collection: {collectionName}")
        df = pd.DataFrame(mongo_client[databaseName][collectionName].find())
        logging.info(f"columns Found: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping id column")
            df.drop("_id", axis=1, inplace=True)
        
        logging.info(f"Row and Columns in df: {df.shape}")
        return df
    except Exception as e:
        raise InsuranceException(e, sys)
    

def convert_to_float(df:pd.DataFrame, exclude_col:list) -> pd.DataFrame:
    try:
        for col in df.columns:
            if col not in exclude_col:
                if df[col].dtype != 'O':
                    df[col] = df[col].astype('float')
        return df
    except Exception as e:
        raise InsuranceException(e, sys)
    

def write_report_yaml(file_path, data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(data, f)
    
    except Exception as e:
        raise InsuranceException(e, sys)
    
def save_object(file_path:str, obj:object):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            return dill.dump(obj, file_obj)
    except Exception as e:
        raise InsuranceException(e, sys)


def load_object(file_path:str):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise InsuranceException(e, sys)
    

def save_to_numpy(file_path:str, array:np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            np.save(f, array)
    except Exception as e:
        raise InsuranceException(e, sys)
    

# Load Model
def load_numpy_data(file_path:str) -> np.array:
    try:
        with open(file_path, "rb") as f:
            return np.load(f)
    except Exception as e:
        raise InsuranceException(e, sys)