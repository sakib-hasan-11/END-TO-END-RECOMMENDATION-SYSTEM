from common.logger import get_logger
from configs.base import BaseConfig
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = get_logger(__name__)


class RankingFeatureBuilder:
    def __init__(self, config: BaseConfig):
        self.config = config

    def build(
        self,
        interactions_df: DataFrame,
        user_features_df: DataFrame,
        item_features_df: DataFrame,
    ) -> DataFrame:
        interactions_df = interactions_df.cache()

        item_features_df = item_features_df.select(
            "article_id",
            "item_purchase_count",
            "avg_item_price",
            "days_since_last_sale",
            "embedding_available",
        )

        user_features_df = user_features_df.select(
            "customer_id",
            "purchase_count",
            "avg_spend",
            "category_diversity",
            "avg_purchase_gap_days",
        )

        print("INTERACTIONS")
        print(interactions_df.columns)

        print("USER")
        print(user_features_df.columns)

        print("ITEM")
        print(item_features_df.columns)

        df = interactions_df.join(user_features_df, on="customer_id", how="inner").join(
            item_features_df, on="article_id", how="inner"
        )

        print(df.columns)
        print(interactions_df.columns)

        from collections import Counter

        counts = Counter(df.columns)

        duplicates = [col_name for col_name, count in counts.items() if count > 1]

        print("DUPLICATES:", duplicates)

        df = df.withColumn(
            "price_affinity",
            F.when(
                F.col("avg_spend") > 0, F.col("avg_item_price") / F.col("avg_spend")
            ).otherwise(0.0),
        )

        df = (
            df.withColumn("day_of_week", F.dayofweek("event_timestamp"))
            .withColumn("month", F.month("event_timestamp"))
            .withColumn(
                "is_weekend",
                F.when(F.dayofweek("event_timestamp").isin([1, 7]), 1).otherwise(0),
            )
        )

        final_df = df.select(
            "customer_id",
            "article_id",
            "embedding_available",
            "purchase_count",
            "avg_spend",
            "category_diversity",
            "avg_purchase_gap_days",
            "item_purchase_count",
            "unique_buyers",
            "avg_item_price",
            "days_since_last_sale",
            "price_affinity",
            "sales_channel_id",
            "day_of_week",
            "month",
            "is_weekend",
        )

        return final_df
