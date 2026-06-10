import pickle
import tempfile
from pathlib import Path

import boto3
import tensorflow as tf
from common.s3_reader import S3Reader
from configs import get_config
from data.loaders import DataLoader
from data.lookup_builder import LookupBuilder
from data.tf_dataset import DatasetBuilder


class TwoTowerDatasetBuilder:
    def __init__(
        self,
        environment: str = "prod",
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

        print("Building artifacts...")

        artifacts = self.lookup_builder.build(
            user_table=user_table,
            item_table=item_table,
            embedding_table=embedding_table,
        )

        print("Building dataset...")

        dataset_builder = DatasetBuilder(artifacts)

        dataset = dataset_builder.build_retrieval_dataset(
            interactions_df,
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

        prefix = f"prepared_datasets/{self.config.environment}/two_tower"

        s3 = boto3.client("s3")

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "tf_dataset"

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

        print("Dataset uploaded.")


def main():
    builder = TwoTowerDatasetBuilder(environment="prod")

    builder.build()


if __name__ == "__main__":
    main()



# python -m training.build_two_tower_dataset
