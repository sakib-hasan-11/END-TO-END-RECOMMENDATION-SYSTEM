from configs.base import BaseConfig


class ProdConfig(BaseConfig):

    def __init__(self):

        super().__init__(
            environment="prod",

            aws_region="us-east-1",

            s3_bucket="recommendation-system-1149",

            ranking_input_path="features/prod/ranking",
            two_tower_input_path="features/prod/two_tower",
            embeddings_input_path="features/prod/embeddings",

            output_root_path="processed/prod",

            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,

            image_embedding_dim=512,

            random_seed=42,

            spark_app_name="recommendation-feature-prod",

            output_partitions=200,
            timestamp_column="event_timestamp"
        )