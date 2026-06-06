import numpy as np
import pandas as pd
import tensorflow as tf

from data.lookup_builder import LookupArtifacts


class RankingDatasetBuilder:
    def __init__(
        self,
        artifacts: LookupArtifacts,
    ):
        self.artifacts = artifacts

    def create_negative_samples(
        self,
        interactions_df: pd.DataFrame,
        negative_ratio: int = 3,
    ) -> pd.DataFrame:
        all_articles = list(self.artifacts.article_to_idx.keys())

        negatives = []

        grouped = interactions_df.groupby("customer_id")

        for customer_id, group in grouped:
            positive_items = set(group["article_id"])

            num_negatives = len(group) * negative_ratio

            sampled_items = np.random.choice(
                all_articles,
                size=num_negatives,
                replace=True,
            )

            for article_id in sampled_items:
                if article_id in positive_items:
                    continue

                negatives.append(
                    {
                        "customer_id": customer_id,
                        "article_id": article_id,
                        "label": 0,
                    }
                )

        return pd.DataFrame(negatives)

    def create_positive_samples(
        self,
        interactions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        positives = interactions_df.copy()

        positives["label"] = 1

        return positives

    def build_training_dataframe(
        self,
        interactions_df: pd.DataFrame,
        negative_ratio: int = 3,
    ):
        positives = self.create_positive_samples(interactions_df)

        negatives = self.create_negative_samples(
            interactions_df,
            negative_ratio,
        )

        ranking_df = pd.concat(
            [
                positives,
                negatives,
            ],
            ignore_index=True,
        )

        ranking_df = ranking_df.sample(
            frac=1,
            random_state=42,
        ).reset_index(drop=True)

        return ranking_df

    def build_tf_dataset(
        self,
        ranking_df: pd.DataFrame,
        batch_size: int = 256,
    ):
        ranking_df["article_id"] = ranking_df["article_id"].astype(str).str.zfill(10)

        embeddings = ranking_df["article_id"].map(
            self.artifacts.article_embedding_lookup
        )

        valid_mask = embeddings.notna()

        ranking_df = ranking_df.loc[valid_mask].reset_index(drop=True)

        embeddings = embeddings.loc[valid_mask].reset_index(drop=True)

        image_embeddings = np.stack(embeddings.values).astype(np.float32)

        user_numeric = ranking_df[
            [
                "purchase_count",
                "avg_spend",
            ]
        ].astype(np.float32)

        item_numeric = ranking_df[
            [
                "item_purchase_count",
                "unique_buyers",
                "avg_item_price",
            ]
        ].astype(np.float32)

        labels = ranking_df["label"].astype(np.float32)

        dataset = (
            tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "user_numeric": user_numeric.values,
                        "item_numeric": item_numeric.values,
                        "image_embedding": image_embeddings,
                    },
                    labels.values,
                )
            )
            .shuffle(10000)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return dataset
