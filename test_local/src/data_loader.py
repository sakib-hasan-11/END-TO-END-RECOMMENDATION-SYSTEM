from pathlib import Path

import boto3

from src.logger import get_logger

logger = get_logger(__name__)


class S3DataLoader:
    def __init__(self, config):
        logger.info(f"Initializing S3DataLoader with bucket: {config.bucket}")
        self.config = config

        try:
            self.s3 = boto3.client("s3")
            logger.info("S3 client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise

    def list_zip_files(self):
        logger.info(
            f"Listing zip files from s3://{self.config.bucket}/{self.config.input_zip_prefix}"
        )
        try:
            paginator = self.s3.get_paginator("list_objects_v2")

            files = []

            for page in paginator.paginate(
                Bucket=self.config.bucket, Prefix=self.config.input_zip_prefix
            ):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".zip"):
                        files.append(obj["Key"])

            logger.info(f"Found {len(files)} zip files in S3")
            return files
        except Exception as e:
            logger.error(f"Failed to list zip files: {str(e)}")
            raise

    def download_zip(self, s3_key):
        logger.info(f"Downloading zip file: {s3_key}")
        try:
            Path(self.config.local_zip_dir).mkdir(parents=True, exist_ok=True)

            local_path = f"{self.config.local_zip_dir}/{Path(s3_key).name}"

            self.s3.download_file(self.config.bucket, s3_key, local_path)

            file_size = Path(local_path).stat().st_size / (1024 * 1024)  # MB
            logger.info(
                f"Successfully downloaded zip to {local_path} (Size: {file_size:.2f} MB)"
            )
            return local_path
        except Exception as e:
            logger.error(f"Failed to download zip file {s3_key}: {str(e)}")
            raise
