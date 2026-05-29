from pyspark.sql import DataFrame

from logs import logger


class DataWriter:
    def __init__(self, test: bool = False):
        self.test = test
        if test:
            self.BASE_PATH = (
                "s3://recommendation-system-1149/processed-data/sample_data"
            )
        else:
            self.BASE_PATH = "s3://recommendation-system-1149/processed-data/main_data"
        logger.info(f"DataWriter initialized: {self.BASE_PATH}")

    def write_articles(self, df: DataFrame, name: str = "articles") -> None:
        try:
            output_path = f"{self.BASE_PATH}/{name}.parquet"
            df.write.mode("overwrite").parquet(output_path)
            logger.info(f"Articles written: {df.count()} records")
        except Exception as e:
            logger.error(f"Error writing articles: {str(e)}")
            raise

    def write_customers(self, df: DataFrame, name: str = "customers") -> None:
        try:
            output_path = f"{self.BASE_PATH}/{name}.parquet"
            df.write.mode("overwrite").parquet(output_path)
            logger.info(f"Customers written: {df.count()} records")
        except Exception as e:
            logger.error(f"Error writing customers: {str(e)}")
            raise

    def write_transactions(self, df: DataFrame, name: str = "transactions") -> None:
        try:
            output_path = f"{self.BASE_PATH}/{name}.parquet"
            df.write.mode("overwrite").parquet(output_path)
            logger.info(f"Transactions written: {df.count()} records")
        except Exception as e:
            logger.error(f"Error writing transactions: {str(e)}")
            raise

    def write_features(
        self, df: DataFrame, feature_type: str) -> None:
        try:
            if self.test:
                output_path = f"s3://recommendation-system-1149/features/sample/{feature_type}"
            else:
                output_path = f"s3://recommendation-system-1149/features/main/{feature_type}"
            df.write.mode("overwrite").parquet(output_path)
            logger.info(f"Features written: {df.count()} records")
            logger.info(f"Successfully written to {output_path}")
        except Exception as e:
            logger.error(f"Error writing features: {str(e)}")
            raise

    def write_custom(self, df: DataFrame, output_path: str) -> None:
        try:
            df.write.mode("overwrite").parquet(output_path)
            logger.info(f"Data written to {output_path}: {df.count()} records")
        except Exception as e:
            logger.error(f"Error writing data: {str(e)}")
            raise
