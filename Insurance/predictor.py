import numpy as np
import pandas as pd
import os
import sys
from typing import Optional
from Insurance.exception import InsuranceException
from Insurance.entity.config_entity import MODEL_FILE_NAME, TARGET_ENCODER_OBJECT_FILE_NAME, TRANSFORMER_OBJECT_FILE_NAME


class SavedModels:
    def __init__(self, model_registry:str = "saved_models",
                 transformer_dir_name:str = "transformer",
                 target_encoder_dir_name: str = "target_encoder",
                 model_dir: str = "model"):
        try:
            self.model_registry = model_registry
            os.makedirs(self.model_registry, exist_ok=True)
            self.transformer_dir_name = transformer_dir_name
            self.target_encoder_dir_name = target_encoder_dir_name
            self.model_dir = model_dir
        except Exception as e:
            raise e
    
    def get_latest_dir_path(self)->Optional[str]:
        try:
            dir_name = os.listdir(self.model_registry)
            if len(dir_name) == 0:
                return None
            dir_name = list(map(int, dir_name))
            latest_dir_name = max(dir_name)
            return os.path.join(self.model_registry, f"{latest_dir_name}")
        
        except Exception as e:
            raise e
        
    def get_latest_model_path(self):
        try:
            latest_dir_path = self.get_latest_dir_path()
            if len(latest_dir_path) == 0:
                raise Exception(f"Model is not available!")
            return os.path.join(latest_dir_path, self.model_dir, MODEL_FILE_NAME)
        
        except Exception as e:
            raise e


    def get_latest_transformer_path(self):
        try:
            latest_dir_path = self.get_latest_dir_path()
            if len(latest_dir_path) == 0:
                raise Exception(f"Transform data path not found!")
            return os.path.join(latest_dir_path, self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)
        
        except Exception as e:
            raise e


    def get_latest_target_encoder_path(self):
        try:
            latest_dir_path = self.get_latest_dir_path()
            if len(latest_dir_path) == 0:
                raise Exception(f"Target Encoder data path not found!")
            return os.path.join(latest_dir_path, self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
        
        except Exception as e:
            raise e
        

    def get_latest_save_dir_path(self)->str:
        try: 
            latest_dir_path = self.get_latest_dir_path()
            if len(latest_dir_path) == 0:
                return os.path.join(self.model_registry, f"{0}")

            latest_dir_number = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registry, f"{latest_dir_number + 1}")   
        except Exception as e:
            raise e 
        

    # Saving Model
    def get_latest_saved_model_path(self):
        try:
            latest_dir_path = self.get_latest_save_dir_path()
            return os.path.join(latest_dir_path, self.model_dir, MODEL_FILE_NAME)
        except Exception as e:
            raise e
        
    # Saving Transformed Data
    def get_latest_saved_trasnformed_path(self):
        try:
            latest_dir_path = self.get_latest_save_dir_path()
            return os.path.join(latest_dir_path, self.transformer_dir_name, TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e
        
    # Saving Target Encoder Data
    def get_latest_saved_target_encoder_path(self):
        try:
            latest_dir_path = self.get_latest_save_dir_path()
            return os.path.join(latest_dir_path, self.target_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e


