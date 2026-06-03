from common.logger import get_logger
from configs.base import BaseConfig
from pyspark.sql import DataFrame, SparkSession

logger = get_logger(__name__)


class S3Reader:
    def __init__(self, spark: SparkSession, config: BaseConfig):
        self.spark = spark
        self.config = config

    def read_parquet(self, relative_path: str) -> DataFrame:
        full_path = f"s3://{self.config.s3_bucket}/{relative_path}"

        try:
            logger.info(f"Reading parquet file: {full_path}")
            df = self.spark.read.parquet(full_path)

            row_count = df.count()
            col_count = len(df.columns)
            logger.info(
                f"Successfully loaded parquet file: rows={row_count}, columns={col_count}"
            )

            return df

        except Exception as e:
            logger.error(
                f"Failed to read parquet file {full_path}: {str(e)}", exc_info=True
            )
            raise

    def load_ranking_features(self):
        return self.read_parquet(self.config.ranking_input_path)

    def load_two_tower_features(self):
        return self.read_parquet(self.config.two_tower_input_path)

    def load_embeddings(self):
        return self.read_parquet(self.config.embeddings_input_path)
