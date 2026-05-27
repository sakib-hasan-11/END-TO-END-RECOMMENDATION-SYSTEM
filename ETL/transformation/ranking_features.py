from pyspark.sql import functions as F
from pyspark.sql.window import Window

from logs import logger


def build_ranking_features(transactions, customers, articles, test: bool = False):
    logger.info("Building ranking model features")

    try:
        if not transactions or transactions.count() == 0:
            raise ValueError("Transactions dataframe is empty")
        if not customers or customers.count() == 0:
            raise ValueError("Customers dataframe is empty")
        if not articles or articles.count() == 0:
            raise ValueError("Articles dataframe is empty")

        df = transactions.join(customers, on="customer_id", how="left").join(
            articles, on="article_id", how="left"
        )
        logger.info(f"Joined data: {df.count()} records")

        df = df.withColumn("event_timestamp", F.to_timestamp("t_dat")).withColumn(
            "event_timestamp", F.col("event_timestamp").cast("timestamp")
        )

        user_stats = df.groupBy("customer_id").agg(
            F.avg("price").alias("user_avg_price"),
            F.count("*").alias("user_total_interactions"),
            F.countDistinct("product_group_name").alias("user_category_diversity"),
        )

        item_stats = df.groupBy("article_id").agg(
            F.count("*").alias("item_popularity"),
            F.avg("price").alias("item_avg_price"),
            F.countDistinct("customer_id").alias("unique_buyers"),
        )

        ranking_df = df.join(user_stats, on="customer_id", how="left").join(
            item_stats, on="article_id", how="left"
        )

        ranking_df = (
            ranking_df.withColumn("day_of_week", F.dayofweek("event_timestamp"))
            .withColumn("month", F.month("event_timestamp"))
            .withColumn("year", F.year("event_timestamp"))
            .withColumn(
                "is_weekend",
                F.when(F.dayofweek("event_timestamp").isin([1, 7]), 1).otherwise(0),
            )
        )

        ranking_df = ranking_df.withColumn(
            "price_affinity", F.abs(F.col("user_avg_price") - F.col("price"))
        ).withColumn(
            "popularity_price_score", F.col("item_popularity") * F.col("price")
        )

        user_window = Window.partitionBy("customer_id")
        ranking_df = ranking_df.withColumn(
            "last_purchase_ts", F.max("event_timestamp").over(user_window)
        ).withColumn(
            "days_since_last_purchase",
            F.datediff(F.col("last_purchase_ts"), F.col("event_timestamp")),
        )

        ranking_df = ranking_df.withColumn("label", F.lit(1))

        if test:
            output_path = (
                "s3://recommendation-system-1149/features/sample/ranking_model/"
            )
        else:
            output_path = "s3://recommendation-system-1149/features/main/ranking_model/"

        ranking_df.write.mode("overwrite").parquet(output_path)
        logger.info(f"Ranking features written to S3: {ranking_df.count()} records")

        return ranking_df

    except Exception as e:
        logger.error(f"Error building ranking features: {str(e)}")
        raise
