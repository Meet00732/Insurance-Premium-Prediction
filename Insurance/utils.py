import numpy as np
import pandas as pd
import pymongo
import os
import sys
from Insurance.exception import InsuranceException
from Insurance.config import mongo_client
from Insurance.logger import logging

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
    
