import os
import pickle
import tempfile
from pathlib import Path

import boto3
from common.s3_reader import S3Reader
from configs import get_config
from data.loaders import DataLoader
from data.lookup_builder import LookupBuilder
from data.ranking_dataset import RankingDatasetBuilder

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf


class RankingDatasetPreparer:
    def __init__(
        self,
        environment="prod",
    ):
        self.config = get_config(environment)

        self.reader = S3Reader()

        self.loader = DataLoader(
            reader=self.reader,
            config=self.config,
        )

        self.lookup_builder = LookupBuilder()

    def build(self):
        print("Loading data...")

        user_table = self.loader.load_user_features()

        item_table = self.loader.load_item_features()

        embedding_table = self.loader.load_image_embeddings()

        interactions = self.loader.load_two_tower_interactions()

        interactions_df = interactions.to_pandas()

        print("Building lookup artifacts...")

        artifacts = self.lookup_builder.build(
            user_table=user_table,
            item_table=item_table,
            embedding_table=embedding_table,
        )

        print("Building ranking dataframe...")

        ranking_builder = RankingDatasetBuilder(artifacts)

        ranking_df = ranking_builder.build_training_dataframe(
            interactions_df,
            negative_ratio=3,
        )

        print(f"Training Rows: {len(ranking_df):,}")

        dataset = ranking_builder.build_tf_dataset(
            ranking_df,
            batch_size=256,
        )

        self.save_to_s3(
            dataset,
            artifacts,
        )

    def save_to_s3(
        self,
        dataset,
        artifacts,
    ):
        bucket = "recommendation-system-1149"

        prefix = f"prepared_datasets/{self.config.environment}/ranking"

        s3 = boto3.client("s3")

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset"

            tf.data.Dataset.save(
                dataset,
                str(dataset_path),
            )

            for file in dataset_path.rglob("*"):
                if file.is_file():
                    s3.upload_file(
                        str(file),
                        bucket,
                        f"{prefix}/dataset/{file.relative_to(dataset_path)}",
                    )

            artifact_file = Path(tmp_dir) / "lookup_artifacts.pkl"

            with open(
                artifact_file,
                "wb",
            ) as f:
                pickle.dump(
                    artifacts,
                    f,
                )

            s3.upload_file(
                str(artifact_file),
                bucket,
                f"{prefix}/lookup_artifacts.pkl",
            )

        print("Uploaded ranking dataset to S3")


def main():
    builder = RankingDatasetPreparer(environment="prod")

    builder.build()


if __name__ == "__main__":
    main()


# python -m training.build_ranking_dataset
