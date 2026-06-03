from dataclasses import dataclass
from typing import Dict


@dataclass
class BaseConfig:

    # Environment
    environment: str

    # AWS
    aws_region: str
    s3_bucket: str

    # Input Data
    ranking_input_path: str
    two_tower_input_path: str
    embeddings_input_path: str

    # Output Data
    output_root_path: str

    # Split Ratios
    train_ratio: float
    validation_ratio: float
    test_ratio: float

    # Embeddings
    image_embedding_dim: int

    # Randomness
    random_seed: int

    # Spark
    spark_app_name: str

    # Partitions
    output_partitions: int

    # Date Column
    timestamp_column: str

    def validate(self):

        total = (
            self.train_ratio +
            self.validation_ratio +
            self.test_ratio
        )

        if round(total, 2) != 1.00:
            raise ValueError(
                f"Split ratios must sum to 1.0. Got {total}"
            )