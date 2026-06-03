from common.logger import get_logger
from configs.base import BaseConfig
from pyspark.sql import DataFrame

logger = get_logger(__name__)


class S3Writer:
    def __init__(self, config: BaseConfig):
        self.config = config

    def write_parquet(self, df: DataFrame, relative_path: str, mode: str = "overwrite"):
        output_path = (
            f"s3://{self.config.s3_bucket}/"
            f"{self.config.output_root_path}/"
            f"{relative_path}"
        )

        try:
            logger.info(
                f"Writing parquet file to {output_path} (mode={mode}, partitions={self.config.output_partitions})"
            )

            (
                df.repartition(self.config.output_partitions)
                .write.mode(mode)
                .parquet(output_path)
            )

            logger.info(f"Successfully wrote parquet file: {output_path}")

        except Exception as e:
            logger.error(
                f"Failed to write parquet file to {output_path}: {str(e)}",
                exc_info=True,
            )
            raise
