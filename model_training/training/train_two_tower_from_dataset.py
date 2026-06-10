import os
import pickle
import tempfile
from pathlib import Path

import boto3
import pandas as pd
import tensorflow as tf
from configs import get_config
from data.lookup_builder import LookupArtifacts
from data.tf_dataset import DatasetBuilder
from models.two_tower import TwoTowerModel

from training.mlflow_tracker import MLflowTracker

os.environ["TF_USE_LEGACY_KERAS"] = "1"


class TwoTowerTrainer:
    def __init__(
        self,
        environment: str = "prod",
    ):
        self.config = get_config(environment)

        self.bucket = "recommendation-system-1149"

        self.dataset_prefix = f"prepared_datasets/{self.config.environment}/two_tower"

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
        s3 = boto3.client("s3")

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_file = Path(tmp_dir) / "lookup_artifacts.pkl"

            s3.download_file(
                self.bucket,
                f"{self.dataset_prefix}/lookup_artifacts.pkl",
                str(local_file),
            )

            with open(
                local_file,
                "rb",
            ) as f:
                artifacts = pickle.load(f)

        return artifacts

    def load_training_chunks(self):
        s3 = boto3.client("s3")

        response = s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=f"{self.dataset_prefix}/dataset",
        )

        chunk_files = []

        for obj in response.get(
            "Contents",
            [],
        ):
            key = obj["Key"]

            if key.endswith(".parquet"):
                chunk_files.append(key)

        chunk_files.sort()

        return chunk_files

    def build_model(
        self,
        artifacts,
    ):
        return TwoTowerModel(
            category_vocab=artifacts.favorite_category_vocab,
            color_vocab=artifacts.favorite_color_vocab,
        )

    def save_model_to_s3(
        self,
        model,
    ):
        s3 = boto3.client("s3")

        prefix = f"model_artifacts/{self.config.environment}/two_tower"

        with tempfile.TemporaryDirectory() as tmp_dir:
            keras_path = Path(tmp_dir) / "two_tower_model.keras"

            model.save(
                keras_path,
            )

            s3.upload_file(
                str(keras_path),
                self.bucket,
                f"{prefix}/two_tower_model.keras",
            )

            saved_model_path = Path(tmp_dir) / "saved_model"

            model.export(str(saved_model_path))

            for file in saved_model_path.rglob("*"):
                if file.is_file():
                    s3.upload_file(
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

        model = self.build_model(artifacts)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

        tracker = MLflowTracker(experiment_name="two_tower_retrieval")

        tracker.start_run(run_name=f"{self.config.environment}_run")

        tracker.log_params(
            {
                "environment": self.config.environment,
                "epochs": 5,
                "learning_rate": 1e-3,
                "chunk_count": len(chunk_files),
            }
        )

        dataset_builder = DatasetBuilder(artifacts)

        final_loss = None

        for idx, chunk_key in enumerate(chunk_files):
            print(f"\nTraining chunk {idx + 1}/{len(chunk_files)}")

            df = pd.read_parquet(f"s3://{self.bucket}/{chunk_key}")

            dataset = dataset_builder.build_retrieval_dataset(
                df,
                batch_size=256,
            )

            history = model.fit(
                dataset,
                epochs=5,
                verbose=1,
            )

            final_loss = float(history.history["loss"][-1])

        tracker.log_metrics(
            {
                "final_loss": final_loss,
            }
        )

        tracker.log_tensorflow_model(
            model,
            artifact_path="two_tower_model",
        )

        self.save_model_to_s3(model)

        tracker.end_run()

        return model


def main():
    trainer = TwoTowerTrainer(environment="prod")

    trainer.train()


if __name__ == "__main__":
    main()


# python -m training.train_two_tower_from_dataset
