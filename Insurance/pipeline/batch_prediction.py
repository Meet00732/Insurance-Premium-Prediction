import sys
import os
import numpy as np
import pandas as pd
from Insurance.entity import artifact_entity, config_entity
from Insurance.exception import InsuranceException
from Insurance import utils
from Insurance.logger import logging
from sklearn.metrics import r2_score
from Insurance.predictor import ModelRegistry
from datetime import datetime

PREDICTION_DIR = "prediction"

def batch_prediction(file_path: str):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        
        # Creating ModelRegistry object to get model, transformer, target_encoder from saved models
        logging.info(f"Creating ModelRegistry object to get model, transformer, target_encoder from saved models")
        model_registry = ModelRegistry(model_registry="saved_models")

        # Read Data to predict
        logging.info(f"Loading data for prediction")
        df = pd.read_csv(file_path)
        logging.info(f"Replace na values with np.Nan")
        df.replace({"na": np.NAN}, inplace=True)

        # Loading Transformer, Model and Target Encoder
        logging.info(f"Loading lastest Transformer, Model nad Target Encoder for Batch Prediction")
        transformer = utils.load_object(model_registry.get_latest_transformer_path())
        model = utils.load_object(model_registry.get_latest_model_path())
        target_encoder = utils.load_object(model_registry.get_latest_target_encoder_path())

        logging.info(f"Performing Feature Encoding and Transformation")
        feature_names = list(transformer.feature_names_in_)
        for i in feature_names:
            if df[i].dtype == 'O':
                df[i] = target_encoder.fit_transform(df[i])

        df_arr = transformer.transform(df[feature_names])
        
        logging.info(f"Performing batch prediction")
        y_pred = model.predict(df_arr)

        df['prediction'] = y_pred

        prediction_file_name = os.path.basename(file_path).replace(".csv", f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_name = os.path.join(PREDICTION_DIR, prediction_file_name)
        df.to_csv(prediction_file_name, index = False, header = True)

        return prediction_file_name
    
    except Exception as e:
        raise InsuranceException(e, sys)