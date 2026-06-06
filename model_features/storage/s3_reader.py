from common.logger import get_logger
from configs.base import BaseConfig
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

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

            #
            # Normalize IDs
            #

            if "article_id" in df.columns:
                df = df.withColumn(
                    "article_id",
                    F.lpad(
                        F.col("article_id").cast("string"),
                        10,
                        "0"
                    )
                )

            if "customer_id" in df.columns:
                df = df.withColumn(
                    "customer_id",
                    F.col("customer_id").cast("string")
                )

            row_count = df.count()
            col_count = len(df.columns)

            logger.info(
                f"Successfully loaded parquet file: "
                f"rows={row_count}, columns={col_count}"
            )

            if "article_id" in df.columns:
                logger.info(
                    f"article_id type = "
                    f"{df.schema['article_id'].dataType}"
                )

            if "customer_id" in df.columns:
                logger.info(
                    f"customer_id type = "
                    f"{df.schema['customer_id'].dataType}"
                )

            return df

        except Exception as e:
            logger.error(
                f"Failed to read parquet file {full_path}: {str(e)}",
                exc_info=True,
            )
            raise

    def load_ranking_features(self):
        return self.read_parquet(
            self.config.ranking_input_path
        )

    def load_two_tower_features(self):
        return self.read_parquet(
            self.config.two_tower_input_path
        )

    def load_embeddings(self):
        return self.read_parquet(
            self.config.embeddings_input_path
        )