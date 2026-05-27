from pyspark.sql import DataFrame, SparkSession

from logs import logger


class DataReader:
    def __init__(self, test: bool = False):
        self.test = test
        if test:
            self.BASE_PATH = "s3://recommendation-system-1149/raw-data/sample_data"
        else:
            self.BASE_PATH = "s3://recommendation-system-1149/raw-data/main_data"
        logger.info(f"DataReader initialized: {self.BASE_PATH}")

    def read_articles(self, spark: SparkSession) -> DataFrame:
        try:
            df = spark.read.parquet(f"{self.BASE_PATH}/articles.parquet")
            logger.info(f"Articles loaded: {df.count()} records")
            return df
        except Exception as e:
            logger.error(f"Error reading articles: {str(e)}")
            raise

    def read_customers(self, spark: SparkSession) -> DataFrame:
        try:
            df = spark.read.parquet(f"{self.BASE_PATH}/customers.parquet")
            logger.info(f"Customers loaded: {df.count()} records")
            return df
        except Exception as e:
            logger.error(f"Error reading customers: {str(e)}")
            raise

    def read_transactions(self, spark: SparkSession) -> DataFrame:
        try:
            df = spark.read.parquet(f"{self.BASE_PATH}/transactions.parquet")
            logger.info(f"Transactions loaded: {df.count()} records")
            return df
        except Exception as e:
            logger.error(f"Error reading transactions: {str(e)}")
            raise

    def read_all(self, spark: SparkSession) -> tuple:
        try:
            articles = self.read_articles(spark)
            customers = self.read_customers(spark)
            transactions = self.read_transactions(spark)
            logger.info("All data loaded successfully")
            return articles, customers, transactions
        except Exception as e:
            logger.error(f"Error reading data: {str(e)}")
            raise
