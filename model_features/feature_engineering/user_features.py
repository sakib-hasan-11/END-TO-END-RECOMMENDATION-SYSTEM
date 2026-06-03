from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from configs.base import BaseConfig


class UserFeatureBuilder:

    def __init__(self, config: BaseConfig):
        self.config = config

    def build(
        self,
        interactions_df: DataFrame
    ) -> DataFrame:

        
        # Base Aggregations
        

        user_agg = (
            interactions_df
            .groupBy("customer_id")
            .agg(
                F.count("*").alias("purchase_count"),

                F.avg("price")
                .alias("avg_spend"),

                F.avg("is_weekend")
                .alias("weekend_ratio"),

                F.countDistinct(
                    "product_group_name"
                ).alias("category_diversity")
            )
        )

        
        # Preferred Channel
        

        channel_window = Window.partitionBy(
            "customer_id"
        ).orderBy(
            F.desc("channel_count")
        )

        preferred_channel = (
            interactions_df
            .groupBy(
                "customer_id",
                "sales_channel_id"
            )
            .count()
            .withColumnRenamed(
                "count",
                "channel_count"
            )
            .withColumn(
                "rn",
                F.row_number().over(
                    channel_window
                )
            )
            .filter(F.col("rn") == 1)
            .select(
                "customer_id",
                F.col("sales_channel_id")
                .alias("preferred_channel")
            )
        )

        
        # Favorite Category
        

        category_window = Window.partitionBy(
            "customer_id"
        ).orderBy(
            F.desc("category_count")
        )

        favorite_category = (
            interactions_df
            .groupBy(
                "customer_id",
                "product_group_name"
            )
            .count()
            .withColumnRenamed(
                "count",
                "category_count"
            )
            .withColumn(
                "rn",
                F.row_number().over(
                    category_window
                )
            )
            .filter(F.col("rn") == 1)
            .select(
                "customer_id",
                F.col("product_group_name")
                .alias("favorite_category")
            )
        )

        
        # Favorite Color
        

        color_window = Window.partitionBy(
            "customer_id"
        ).orderBy(
            F.desc("color_count")
        )

        favorite_color = (
            interactions_df
            .groupBy(
                "customer_id",
                "colour_group_name"
            )
            .count()
            .withColumnRenamed(
                "count",
                "color_count"
            )
            .withColumn(
                "rn",
                F.row_number().over(
                    color_window
                )
            )
            .filter(F.col("rn") == 1)
            .select(
                "customer_id",
                F.col("colour_group_name")
                .alias("favorite_color")
            )
        )

        
        # Purchase Gap Calculation
        

        purchase_window = Window.partitionBy(
            "customer_id"
        ).orderBy(
            "event_timestamp"
        )

        purchase_gap = (
            interactions_df
            .select(
                "customer_id",
                "event_timestamp"
            )
            .withColumn(
                "previous_purchase",
                F.lag(
                    "event_timestamp"
                ).over(
                    purchase_window
                )
            )
            .withColumn(
                "gap_days",
                F.datediff(
                    F.col("event_timestamp"),
                    F.col("previous_purchase")
                )
            )
        )

        avg_purchase_gap = (
            purchase_gap
            .groupBy("customer_id")
            .agg(
                F.avg("gap_days")
                .alias("avg_purchase_gap_days")
            )
        )

        
        # Final Join
        

        final_df = (
            user_agg
            .join(
                preferred_channel,
                on="customer_id",
                how="left"
            )
            .join(
                favorite_category,
                on="customer_id",
                how="left"
            )
            .join(
                favorite_color,
                on="customer_id",
                how="left"
            )
            .join(
                avg_purchase_gap,
                on="customer_id",
                how="left"
            )
        )

        return final_df