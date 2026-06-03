import json
from typing import Dict

import boto3
from common.logger import get_logger
from configs.base import BaseConfig

logger = get_logger(__name__)


class ArtifactManager:
    def __init__(self, config: BaseConfig):
        self.config = config

        self.s3_client = boto3.client("s3", region_name=config.aws_region)

    def save_json(self, artifact: Dict, artifact_name: str):
        key = f"{self.config.output_root_path}/artifacts/{artifact_name}.json"

        try:
            logger.info(f"Saving artifact {artifact_name} to S3...")

            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=key,
                Body=json.dumps(artifact, indent=4, default=str),
            )

            logger.info(
                f"Successfully saved artifact: s3://{self.config.s3_bucket}/{key}"
            )

        except Exception as e:
            logger.error(
                f"Failed to save artifact {artifact_name}: {str(e)}", exc_info=True
            )
            raise

    def build_training_metadata(
        self, train_count: int, validation_count: int, test_count: int
    ):
        return {
            "environment": self.config.environment,
            "train_rows": train_count,
            "validation_rows": validation_count,
            "test_rows": test_count,
            "embedding_dimension": self.config.image_embedding_dim,
            "random_seed": self.config.random_seed,
        }

    def save_feature_schema(self, schema_dict: Dict):
        self.save_json(schema_dict, "feature_schema")
