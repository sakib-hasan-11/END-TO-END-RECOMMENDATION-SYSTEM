import sys
from datetime import datetime
from typing import Tuple

from pyspark.sql import DataFrame

sys.path.insert(0, "..")

from readers.s3_reader import DataReader
from transformation.pre_process import preprocess
from transformation.ranking_features import build_ranking_features
from transformation.two_tower_featues import build_two_tower_features
from utils.spark_session import create_spark_session
from writers.s3_writer import DataWriter

from logs import logger


class ETLPipeline:
    def __init__(
        self, test: bool = False, app_name: str = "recommendation-etl-pipeline"
    ):
        self.test = test
        self.app_name = app_name
        self.spark = None
        self.reader = None
        self.writer = None
        self.start_time = None
        logger.info(
            f"ETL Pipeline initialized: {'test' if test else 'production'} mode"
        )

    def setup(self) -> None:
        try:
            self.spark = create_spark_session(app_name=self.app_name)
            self.reader = DataReader(test=self.test)
            self.writer = DataWriter(test=self.test)
            self.start_time = datetime.now()
            logger.info("Pipeline setup completed")
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            raise

    def read_data(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        try:
            logger.info("Reading raw data from S3")
            articles, customers, transactions = self.reader.read_all(self.spark)

            if (
                articles.count() == 0
                or customers.count() == 0
                or transactions.count() == 0
            ):
                raise ValueError("One or more input dataframes are empty")

            logger.info(
                f"Data read: articles {articles.count()}, customers {customers.count()}, transactions {transactions.count()}"
            )
            return articles, customers, transactions
        except Exception as e:
            logger.error(f"Read failed: {str(e)}")
            raise

    def preprocess_data(
        self, customers: DataFrame, articles: DataFrame, transactions: DataFrame
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        try:
            logger.info("Preprocessing data")
            customers, articles, transactions = preprocess(
                customers=customers,
                articles=articles,
                transactions=transactions,
                test=self.test,
            )
            logger.info("Preprocessing completed")
            return customers, articles, transactions
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def build_ranking_features(
        self, customers: DataFrame, articles: DataFrame, transactions: DataFrame
    ) -> DataFrame:
        try:
            logger.info("Building ranking model features")
            ranking_df = build_ranking_features(
                transactions=transactions,
                customers=customers,
                articles=articles,
                test=self.test,
            )
            logger.info(f"Ranking features built: {ranking_df.count()} records")
            return ranking_df
        except Exception as e:
            logger.error(f"Ranking features failed: {str(e)}")
            raise

    def build_two_tower_features(
        self, customers: DataFrame, articles: DataFrame, transactions: DataFrame
    ) -> DataFrame:
        try:
            logger.info("Building two-tower model features")
            two_tower_df = build_two_tower_features(
                transactions=transactions,
                customers=customers,
                articles=articles,
                test=self.test,
            )
            logger.info(f"Two-tower features built: {two_tower_df.count()} records")
            return two_tower_df
        except Exception as e:
            logger.error(f"Two-tower features failed: {str(e)}")
            raise

    def run(self) -> None:
        try:
            self.setup()
            articles, customers, transactions = self.read_data()
            customers, articles, transactions = self.preprocess_data(
                customers, articles, transactions
            )
            ranking_df = self.build_ranking_features(customers, articles, transactions)
            two_tower_df = self.build_two_tower_features(
                customers, articles, transactions
            )

            logger.info("Writing ranking features...")
            self.writer.write_features(
                ranking_df,
                feature_type="ranking"
            )

            logger.info("Writing two tower features...")
            self.writer.write_features(
                two_tower_df,
                feature_type="two_tower"
            )


            logger.info("All writes verified")

            end_time = datetime.now()
            duration = end_time - self.start_time
            logger.info(f"Pipeline completed successfully. Duration: {duration}")

            if self.spark:
                self.spark.stop()
                logger.info("Spark session stopped")
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            if self.spark:
                self.spark.stop()
            raise


def main(test: bool = False):
    try:
        pipeline = ETLPipeline(test=test)
        pipeline.run()
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ETL Pipeline for Recommendation System"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "prod"],
        default="test",
        help="Run in test or prod mode",
    )
    args = parser.parse_args()
    main(test=args.mode == "test")
