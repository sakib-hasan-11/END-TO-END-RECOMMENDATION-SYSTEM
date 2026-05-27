from pyspark.sql import Window
from pyspark.sql import functions as F

from logs import logger


def build_customer_features(df):
    logger.info("Building customer features")
    try:
        if not df or df.count() == 0:
            raise ValueError("Input dataframe is empty")
        user_purchase_count = df.groupBy("customer_id").agg(
            F.count("*").alias("purchase_count")
        )
        user_avg_spend = df.groupBy("customer_id").agg(
            F.avg("price").alias("avg_spend")
        )
        favorite_category = (
            df.groupBy("customer_id", "product_group_name")
            .agg(F.count("*").alias("category_count"))
            .withColumn(
                "rn",
                F.row_number().over(
                    Window.partitionBy("customer_id").orderBy(F.desc("category_count"))
                ),
            )
            .filter(F.col("rn") == 1)
            .select(
                "customer_id", F.col("product_group_name").alias("favorite_category")
            )
        )
        recency = df.groupBy("customer_id").agg(F.max("t_dat").alias("last_purchase"))
        customer_features = (
            user_purchase_count.join(user_avg_spend, on="customer_id", how="left")
            .join(favorite_category, on="customer_id", how="left")
            .join(recency, on="customer_id", how="left")
        )
        logger.info(f"Customer features built: {customer_features.count()} records")
        return customer_features
    except Exception as e:
        logger.error(f"Error building customer features: {str(e)}")
        raise


def item_popularity_features(df):
    logger.info("Building item popularity features")
    try:
        if not df or df.count() == 0:
            raise ValueError("Input dataframe is empty")
        item_popularity = df.groupBy("article_id").agg(
            F.count("*").alias("item_purchase_count")
        )
        item_unique_users = df.groupBy("article_id").agg(
            F.countDistinct("customer_id").alias("unique_buyers")
        )
        item_avg_price = df.groupBy("article_id").agg(
            F.avg("price").alias("avg_item_price")
        )
        item_features = item_popularity.join(
            item_unique_users, on="article_id", how="left"
        ).join(item_avg_price, on="article_id", how="left")
        logger.info(f"Item features built: {item_features.count()} records")
        return item_features
    except Exception as e:
        logger.error(f"Error building item popularity features: {str(e)}")
        raise


def temporal_features(df):
    logger.info("Building temporal features")
    try:        
        if not df or df.count() == 0:
            raise ValueError("Input dataframe is empty")
        df = (
            df.withColumn("day_of_week", F.dayofweek("event_timestamp"))
            .withColumn("month", F.month("event_timestamp"))
            .withColumn("year", F.year("event_timestamp"))
            .withColumn(
                "is_weekend",
                F.when(F.dayofweek("event_timestamp").isin([1, 7]), 1).otherwise(0),
            )
        )
        logger.info("Temporal features added")
        return df
    except Exception as e:
        logger.error(f"Error building temporal features: {str(e)}")
        raise
