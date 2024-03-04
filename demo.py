# Batch Prediction
# Training Pipeline

from Insurance.pipeline.batch_prediction import batch_prediction
from Insurance.pipeline.training_pipeline import training_pipeline

FILE_PATH = "insurance.csv"

if __name__ == "__main__":
    try:
        batch_prediction_output = batch_prediction(file_path=FILE_PATH)
        # training_output = training_pipeline()

    except Exception as e:
        raise e