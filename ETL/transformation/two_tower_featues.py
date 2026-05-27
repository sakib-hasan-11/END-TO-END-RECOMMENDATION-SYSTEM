from pyspark.sql import functions as F
from pyspark.sql.window import Window

from logs import logger
from transformation.features import (
    build_customer_features,
    item_popularity_features,
    temporal_features,
)
from transformation.joins import join_all


def build_two_tower_features(transactions, customers, articles, test: bool = False):
    logger.info("Building two-tower model features")

    try:
        if not transactions or transactions.count() == 0:
            raise ValueError("Transactions dataframe is empty")
        if not customers or customers.count() == 0:
            raise ValueError("Customers dataframe is empty")
        if not articles or articles.count() == 0:
            raise ValueError("Articles dataframe is empty")

        df = join_all(customer=customers, transaction=transactions, article=articles)
        logger.info(f"Data joined: {df.count()} records")

        user_features = build_customer_features(df)
        item_features = item_popularity_features(df)

        color_affinity = (
            df.groupBy("customer_id", "colour_group_name")
            .agg(F.count("*").alias("color_interaction_count"))
            .withColumn(
                "color_rank",
                F.row_number().over(
                    Window.partitionBy("customer_id").orderBy(
                        F.desc("color_interaction_count")
                    )
                ),
            )
            .filter(F.col("color_rank") == 1)
            .select("customer_id", F.col("colour_group_name").alias("favorite_color"))
        )

        df = temporal_features(df)

        final_df = (
            df.join(user_features, on="customer_id", how="left")
            .join(item_features, on="article_id", how="left")
            .join(color_affinity, on="customer_id", how="left")
        )

        if test:
            output_path = (
                "s3://recommendation-system-1149/features/sample/two_tower_model/"
            )
        else:
            output_path = (
                "s3://recommendation-system-1149/features/main/two_tower_model/"
            )

        final_df.write.mode("overwrite").parquet(output_path)
        logger.info(f"Two-tower features written to S3: {final_df.count()} records")

        return final_df

    except Exception as e:
        logger.error(f"Error building two-tower features: {str(e)}")
        raise
