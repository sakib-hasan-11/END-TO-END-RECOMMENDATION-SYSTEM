import os
import pickle
import tempfile
from pathlib import Path

import boto3
import pandas as pd

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from configs import get_config
from data.lookup_builder import LookupArtifacts
from data.ranking_dataset import RankingDatasetBuilder
from models.ranking_model import RankingModel

from training.mlflow_tracker import MLflowTracker


class RankingTrainer:
    def __init__(
        self,
        environment="prod",
    ):
        self.config = get_config(environment)

        self.bucket = "recommendation-system-1149"

        self.dataset_prefix = f"prepared_datasets/{self.config.environment}/ranking"

        self.s3 = boto3.client("s3")

    def setup_device(self):
        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(
                        gpu,
                        True,
                    )

                print(f"Using GPU: {len(gpus)} detected")

            except RuntimeError as e:
                print(e)

        else:
            print("Using CPU")

    def load_lookup_artifacts(
        self,
    ) -> LookupArtifacts:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_file = Path(tmp_dir) / "lookup_artifacts.pkl"

            self.s3.download_file(
                self.bucket,
                f"{self.dataset_prefix}/lookup_artifacts.pkl",
                str(local_file),
            )

            with open(local_file, "rb") as f:
                artifacts = pickle.load(f)

        return artifacts

    def load_training_chunks(self):
        paginator = self.s3.get_paginator("list_objects_v2")

        chunk_files = []

        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=f"{self.dataset_prefix}/dataset",
        ):
            for obj in page.get(
                "Contents",
                [],
            ):
                key = obj["Key"]

                if key.endswith(".parquet"):
                    chunk_files.append(key)

        chunk_files.sort()

        return chunk_files

    def download_chunk(
        self,
        chunk_key,
        local_path,
    ):
        self.s3.download_file(
            self.bucket,
            chunk_key,
            str(local_path),
        )

    def build_model(self):
        return RankingModel()

    def save_model_to_s3(
        self,
        model,
    ):
        prefix = f"model_artifacts/{self.config.environment}/ranking"

        with tempfile.TemporaryDirectory() as tmp_dir:
            keras_path = Path(tmp_dir) / "ranking_model.keras"

            model.save(keras_path)

            self.s3.upload_file(
                str(keras_path),
                self.bucket,
                f"{prefix}/ranking_model.keras",
            )

            saved_model_path = Path(tmp_dir) / "saved_model"

            model.export(str(saved_model_path))

            for file in saved_model_path.rglob("*"):
                if file.is_file():
                    self.s3.upload_file(
                        str(file),
                        self.bucket,
                        f"{prefix}/saved_model/{file.relative_to(saved_model_path)}",
                    )

        print(f"Model uploaded to s3://{self.bucket}/{prefix}")

    def train(self):
        self.setup_device()

        print("Loading lookup artifacts...")

        artifacts = self.load_lookup_artifacts()

        print("Loading chunk list...")

        chunk_files = self.load_training_chunks()

        print(f"Found {len(chunk_files)} chunks")

        ranking_builder = RankingDatasetBuilder(artifacts)

        model = self.build_model()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        tracker = MLflowTracker(experiment_name="ranking_model")

        tracker.start_run(run_name=f"{self.config.environment}_run")

        tracker.log_params(
            {
                "environment": self.config.environment,
                "learning_rate": 1e-3,
                "chunk_count": len(chunk_files),
            }
        )

        final_metrics = {}

        #
        # One pass through all chunks = 1 epoch
        #
        for chunk_idx, chunk_key in enumerate(chunk_files):
            print(f"\nTraining chunk {chunk_idx + 1}/{len(chunk_files)}")

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_file = Path(tmp_dir) / "chunk.parquet"

                self.download_chunk(
                    chunk_key,
                    local_file,
                )

                df = pd.read_parquet(local_file)

            dataset = ranking_builder.build_tf_dataset(
                df,
                batch_size=256,
            )

            history = model.fit(
                dataset,
                epochs=1,
                verbose=1,
            )

            final_metrics = {
                "loss": float(history.history["loss"][-1]),
                "auc": float(history.history["auc"][-1]),
                "accuracy": float(history.history["accuracy"][-1]),
                "precision": float(history.history["precision"][-1]),
                "recall": float(history.history["recall"][-1]),
            }

        tracker.log_metrics(final_metrics)

        tracker.log_tensorflow_model(
            model,
            artifact_path="ranking_model",
        )

        self.save_model_to_s3(model)

        tracker.end_run()

        return model


def main():
    trainer = RankingTrainer(environment="test")

    trainer.train()


if __name__ == "__main__":
    main()
