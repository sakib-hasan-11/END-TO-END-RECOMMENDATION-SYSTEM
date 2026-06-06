from configs.base import BaseConfig


class ProdConfig(BaseConfig):
    def __init__(self):
        super().__init__(
            environment="prod",
            processed_root="s3://recommendation-system-1149/processed/prod/train/",
            embeddings_root="s3://recommendation-system-1149/features/main/embeddings/",
            batch_size=8192,
            epochs=10,
            MODEL_ARTIFACT_PATH=(
                "s3://recommendation-system-1149/model_artifacts/prod/"
            ),
        )
