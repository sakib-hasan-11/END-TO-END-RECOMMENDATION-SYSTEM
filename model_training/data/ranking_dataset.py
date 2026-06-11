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
        all_articles = self.artifacts.valid_ranking_articles

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

        positives["article_id"] = positives["article_id"].astype(str).str.zfill(10)

        negatives = self.create_negative_samples(
            interactions_df,
            negative_ratio,
        )

        # feature_cols = [
        #     "purchase_count",
        #     "avg_spend",
        #     "item_purchase_count",
        #     "unique_buyers",
        #     "avg_item_price",
        # ]

        user_features = interactions_df[
            [
                "customer_id",
                "purchase_count",
                "avg_spend",
            ]
        ].drop_duplicates("customer_id")

        # item_features = (
        #     interactions_df[
        #         [
        #             "article_id",
        #             "item_purchase_count",
        #             "unique_buyers",
        #             "avg_item_price",
        #         ]
        #     ]
        #     .drop_duplicates("article_id")
        # )

        # item_features["article_id"] = (
        #     item_features["article_id"]
        #     .astype(str)
        #     .str.zfill(10)
        # )

        item_feature_records = negatives["article_id"].map(
            self.artifacts.item_feature_lookup
        )

        negatives["article_id"] = negatives["article_id"].astype(str).str.zfill(10)

        negatives = negatives.merge(
            user_features,
            on="customer_id",
            how="left",
        )

        # negatives = negatives.merge(
        #     item_features,
        #     on="article_id",
        #     how="left",
        # )

        negatives["item_purchase_count"] = item_feature_records.apply(
            lambda x: x["item_purchase_count"] if x is not None else 0.0
        )

        negatives["unique_buyers"] = item_feature_records.apply(
            lambda x: x["unique_buyers"] if x is not None else 0.0
        )

        negatives["avg_item_price"] = item_feature_records.apply(
            lambda x: x["avg_item_price"] if x is not None else 0.0
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

        ranking_df["article_id"] = ranking_df["article_id"].astype(str).str.zfill(10)

        embeddings = ranking_df["article_id"].map(
            self.artifacts.article_embedding_lookup
        )

        valid_mask = embeddings.notna()

        ranking_df = ranking_df.loc[valid_mask].reset_index(drop=True)

        ranking_df["image_embedding"] = embeddings.loc[valid_mask].reset_index(
            drop=True
        )

        ranking_df = ranking_df[
            [
                "purchase_count",
                "avg_spend",
                "item_purchase_count",
                "unique_buyers",
                "avg_item_price",
                "image_embedding",
                "label",
            ]
        ]

        print(
            ranking_df[
                [
                    "purchase_count",
                    "avg_spend",
                    "item_purchase_count",
                    "unique_buyers",
                    "avg_item_price",
                    "image_embedding",
                    "label",
                ]
            ]
            .isna()
            .sum()
        )

        return ranking_df

    def build_tf_dataset(
        self,
        ranking_df: pd.DataFrame,
        batch_size: int = 256,
    ):
        image_embeddings = np.stack(ranking_df["image_embedding"].values).astype(
            np.float32
        )

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
