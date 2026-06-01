import boto3

from src.logger import get_logger

logger = get_logger(__name__)


class S3Writer:
    def __init__(self, config):
        logger.info(f"Initializing S3Writer with bucket: {config.bucket}")
        self.config = config

        try:
            self.s3 = boto3.client("s3")
            logger.info("S3 client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise

    def upload_file(self, local_path, s3_key):
        logger.info(f"Uploading file to s3://{self.config.bucket}/{s3_key}")
        try:
            self.s3.upload_file(local_path, self.config.bucket, s3_key)
            logger.info(f"Successfully uploaded to S3")

            # Verify upload
            try:
                response = self.s3.head_object(Bucket=self.config.bucket, Key=s3_key)
                logger.debug(
                    f"Upload verified - File size in S3: {response['ContentLength']} bytes"
                )
            except Exception as e:
                logger.warning(f"Could not verify upload: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {str(e)}")
            raise
