from configs.base import BaseConfig


class TestConfig(BaseConfig):
    def __init__(self):
        super().__init__(
            environment="test",
            aws_region="us-east-1",
            s3_bucket="recommendation-system-1149",
            ranking_input_path="features/sample/ranking",
            two_tower_input_path="features/sample/two_tower",
            embeddings_input_path="features/sample/embeddings",
            output_root_path="processed/test",
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
            image_embedding_dim=512,
            random_seed=42,
            spark_app_name="recommendation-feature-test",
            output_partitions=4,
            timestamp_column="event_timestamp",
        )
