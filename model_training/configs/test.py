from configs.base import BaseConfig


class TestConfig(BaseConfig):
    def __init__(self):
        super().__init__(
            environment="test",
            processed_root="s3://recommendation-system-1149/processed/test/train/",
            embeddings_root="s3://recommendation-system-1149/features/sample/embeddings/",
            batch_size=1024,
            epochs=2,
            MODEL_ARTIFACT_PATH=(
                "s3://recommendation-system-1149/model_artifacts/test/"
            ),
        )
