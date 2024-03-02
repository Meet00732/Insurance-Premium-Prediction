from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    train_data_file_path: str
    test_data_file_path: str

