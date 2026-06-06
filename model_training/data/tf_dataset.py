from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from data.lookup_builder import LookupArtifacts


class DatasetBuilder:
    def __init__(
        self,
        artifacts: LookupArtifacts,
    ):
        self.artifacts = artifacts

    def build_retrieval_dataset(
        self,
        interactions_df: pd.DataFrame,
        batch_size: int,
    ) -> tf.data.Dataset:
        interactions_df = interactions_df.copy()

        interactions_df["article_id"] = (
            interactions_df["article_id"].astype(str).str.zfill(10)
        )

        mapped_embeddings = interactions_df["article_id"].map(
            self.artifacts.article_embedding_lookup
        )

        valid_mask = mapped_embeddings.notna()

        interactions_df = interactions_df.loc[valid_mask].reset_index(drop=True)

        mapped_embeddings = mapped_embeddings.loc[valid_mask].reset_index(drop=True)

        print(f"Rows after embedding filter: {len(interactions_df):,}")

        user_numeric = interactions_df[
            [
                "purchase_count",
                "avg_spend",
            ]
        ].astype(np.float32)

        user_category = (
            interactions_df["favorite_category"].fillna("unknown").astype(str)
        )

        user_color = interactions_df["favorite_color"].fillna("unknown").astype(str)

        item_numeric = interactions_df[
            [
                "item_purchase_count",
                "unique_buyers",
                "avg_item_price",
            ]
        ].astype(np.float32)

        item_categorical = interactions_df[
            [
                "department_no",
                "product_type_no",
                "section_no",
                "garment_group_no",
                "colour_group_code",
                "index_group_no",
            ]
        ].astype(np.int32)

        image_embeddings = np.stack(mapped_embeddings.values).astype(np.float32)

        print(user_numeric.shape)
        print(item_numeric.shape)
        print(item_categorical.shape)
        print(image_embeddings.shape)

        dataset = tf.data.Dataset.from_tensor_slices(
            {
                "user_numeric": user_numeric.values,
                "user_category": user_category.values,
                "user_color": user_color.values,
                "item_numeric": item_numeric.values,
                "item_categorical": item_categorical.values,
                "image_embedding": image_embeddings,
            }
        )

        dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset
