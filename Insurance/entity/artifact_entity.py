from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    train_data_file_path: str
    test_data_file_path: str

@dataclass
class DataValidationArtifact:
    validation_report_file: str


@dataclass
class DataTransformationArtifact:
    transform_object_path: str
    transform_train_path: str
    transform_test_path: str
    target_encoder_path: str

@dataclass
class ModelTrainingArtifact:
    model_path: str
    r2_train_score: float
    r2_test_score: float

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_score: float

@dataclass
class ModelPusherArtifact:
    pusher_model_dir: str
    saved_model_dir: str