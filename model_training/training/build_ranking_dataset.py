import gc
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
        environment="test",
    ):
        self.config = get_config(environment)

        self.reader = S3Reader()

        self.loader = DataLoader(
            reader=self.reader,
            config=self.config,
        )

        self.lookup_builder = LookupBuilder()

    def save_lookup_artifacts(
        self,
        artifacts,
    ):
        bucket = "recommendation-system-1149"

        prefix = f"prepared_datasets/{self.config.environment}/ranking"

        s3 = boto3.client("s3")

        with tempfile.TemporaryDirectory() as tmp_dir:
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

        print("Lookup artifacts uploaded")

    def save_dataset_chunk(
        self,
        dataset,
        chunk_id,
    ):
        bucket = "recommendation-system-1149"

        prefix = f"prepared_datasets/{self.config.environment}/ranking"

        s3 = boto3.client("s3")

        with tempfile.TemporaryDirectory() as tmp_dir:
            shard_path = Path(tmp_dir) / f"chunk_{chunk_id:05d}"

            tf.data.Dataset.save(
                dataset,
                str(shard_path),
            )
            print(f"Uploading chunk {chunk_id:05d}")
            for file in shard_path.rglob("*"):
                if file.is_file():
                    s3.upload_file(
                        str(file),
                        bucket,
                        f"{prefix}/dataset/"
                        f"chunk_{chunk_id:05d}/"
                        f"{file.relative_to(shard_path)}",
                    )

    def build(self):
        print("Loading lookup source tables...")

        user_table = self.loader.load_user_features()

        item_table = self.loader.load_item_features()

        embedding_table = self.loader.load_image_embeddings()

        print("Building lookup artifacts...")

        artifacts = self.lookup_builder.build(
            user_table=user_table,
            item_table=item_table,
            embedding_table=embedding_table,
        )

        del user_table
        del item_table
        del embedding_table

        gc.collect()

        self.save_lookup_artifacts(artifacts)

        ranking_builder = RankingDatasetBuilder(artifacts)

        interactions = self.loader.load_two_tower_interactions_streaming()

        fragments = list(interactions.get_fragments())

        # total_rows = interactions.num_rows

        print(f"Found {len(fragments)} parquet fragments")

        # print(f"Total interactions: {total_rows:,}")

        if self.config.environment == "test":
            chunk_size = 100000
        else:
            chunk_size = 250000

        # total_chunks = math.ceil(total_rows / chunk_size)

        # print(f"Processing {total_chunks} chunks")

        scanner = interactions.scanner(batch_size=chunk_size)

        for chunk_id, batch in enumerate(scanner.to_batches()):
            # print(f"\nChunk {chunk_id + 1}/{total_chunks}")

            interactions_df = batch.to_pandas()

            print(f"Loaded {len(interactions_df):,} interactions")

            ranking_df = ranking_builder.build_training_dataframe(
                interactions_df,
                negative_ratio=3,
            )

            print(f"Generated {len(ranking_df):,} ranking rows")

            dataset = ranking_builder.build_tf_dataset(
                ranking_df,
                batch_size=256,
            )

            self.save_dataset_chunk(
                dataset=dataset,
                chunk_id=chunk_id,
            )

            del interactions_df
            del ranking_df
            del dataset

            gc.collect()

            print(f"Chunk {chunk_id + 1} uploaded")

        print("All chunks uploaded")


def main():
    builder = RankingDatasetPreparer(environment="test")

    builder.build()


if __name__ == "__main__":
    main()


# python -m training.build_ranking_dataset
