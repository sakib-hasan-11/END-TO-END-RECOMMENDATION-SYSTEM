import pandas as pd


class TwoTowerDatasetPreparer:
    def __init__(
        self,
        artifacts,
    ):
        self.artifacts = artifacts

    def build_training_dataframe(
        self,
        interactions_df: pd.DataFrame,
    ):
        df = interactions_df.copy()

        df = df[df["article_id"].isin(self.artifacts.article_to_idx)]

        df["article_idx"] = df["article_id"].map(self.artifacts.article_to_idx)

        df["customer_idx"] = df["customer_id"].map(self.artifacts.customer_to_idx)

        embedding_lookup = self.artifacts.article_embedding_lookup

        df["image_embedding"] = df["article_id"].map(embedding_lookup)

        df = df.dropna(subset=["image_embedding"])

        return df.reset_index(drop=True)
