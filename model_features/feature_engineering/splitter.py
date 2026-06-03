from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from datetime import datetime
from configs.base import BaseConfig


class DatasetSplitter:

    def __init__(self, config: BaseConfig):
        self.config = config

    def split(
        self,
        df: DataFrame
    ):

        timestamp_col = self.config.timestamp_column

        min_ts = (
            df
            .agg(F.min(timestamp_col))
            .collect()[0][0]
        )

        max_ts = (
            df
            .agg(F.max(timestamp_col))
            .collect()[0][0]
        )

        total_seconds = (
            max_ts.timestamp()
            - min_ts.timestamp()
        )

        # train_end = (
        #     min_ts.timestamp()
        #     + total_seconds * self.config.train_ratio
        # )

        train_end_ts = datetime.fromtimestamp(
            min_ts.timestamp()
            + total_seconds * self.config.train_ratio
        )

        if min_ts is None or max_ts is None:
            raise ValueError(
                f"No valid timestamps found in {timestamp_col}")

        # validation_end = (
        #     train_end_ts
        #     + total_seconds * self.config.validation_ratio
        # )

        validation_end_ts = datetime.fromtimestamp(
            train_end_ts.timestamp()
            + total_seconds * self.config.validation_ratio
        )

        train_df = df.filter(
            F.col(timestamp_col) <
            F.lit(F.lit(train_end_ts))
        )

        validation_df = df.filter(
            (F.col(timestamp_col) >=
             F.lit(F.lit(train_end_ts)))
            &
            (F.col(timestamp_col) <
             F.lit(F.lit(validation_end_ts)))
        )

        test_df = df.filter(
            F.col(timestamp_col) >=
            F.lit(F.lit(validation_end_ts))
        )

        return (
            train_df,
            validation_df,
            test_df
        )
    # def validate_split(
    # self,
    # train_df,
    # validation_df,
    # test_df):
    #     pass

    # def print_summary(
    #     self,
    #     train_df,
    #     validation_df,
    #     test_df):
    #     pass