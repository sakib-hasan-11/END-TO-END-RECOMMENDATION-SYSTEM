from common.logger import get_logger
from configs.base import BaseConfig
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = get_logger(__name__)


class ItemFeatureBuilder:
    def __init__(self, config: BaseConfig):
        self.config = config

    def build(self, interactions_df: DataFrame, embeddings_df: DataFrame) -> DataFrame:
        interactions_df = interactions_df.cache()

        # Item Aggregations

        item_features = interactions_df.groupBy("article_id").agg(
            F.count("*").alias("item_purchase_count"),
            F.countDistinct("customer_id").alias("unique_buyers"),
            F.avg("price").alias("avg_item_price"),
            F.first("department_no", ignorenulls=True).alias("department_no"),
            F.first("product_type_no", ignorenulls=True).alias("product_type_no"),
            F.first("section_no", ignorenulls=True).alias("section_no"),
            F.first("garment_group_no", ignorenulls=True).alias("garment_group_no"),
            F.first("colour_group_code", ignorenulls=True).alias("colour_group_code"),
            F.first("index_group_no", ignorenulls=True).alias("index_group_no"),
            F.max("event_timestamp").alias("last_sale_timestamp"),
        )

        max_timestamp = interactions_df.agg(F.max("event_timestamp")).collect()[0][0]

        item_features = item_features.withColumn(
            "days_since_last_sale",
            F.datediff(F.lit(max_timestamp), F.col("last_sale_timestamp")),
        )

        # Embedding Cleanup

        embeddings_clean = embeddings_df.select(
            "article_id", "image_embedding"
        ).dropDuplicates(["article_id"])

        # Join Embeddings

        final_df = item_features.join(embeddings_clean, on="article_id", how="left")

        # Embedding Availability Flag

        final_df = final_df.withColumn(
            "embedding_available",
            F.when(F.col("image_embedding").isNotNull(), F.lit(1)).otherwise(F.lit(0)),
        ).drop("last_sale_timestamp")

        return final_df
