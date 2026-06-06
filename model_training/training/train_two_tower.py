import os
import tempfile
from pathlib import Path

import boto3
from common.s3_reader import S3Reader
from configs import get_config
from data.loaders import DataLoader
from data.lookup_builder import LookupBuilder
from data.tf_dataset import DatasetBuilder
from models.two_tower import TwoTowerModel

from training.mlflow_tracker import MLflowTracker

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf


class TwoTowerTrainer:
    def __init__(
        self,
        environment: str = "test",
    ):
        self.config = get_config(environment)

        self.reader = S3Reader()

        self.loader = DataLoader(
            reader=self.reader,
            config=self.config,
        )

        self.lookup_builder = LookupBuilder()

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
            print("No GPU detected. Using CPU.")

    def load_data(self):
        user_table = self.loader.load_user_features()

        item_table = self.loader.load_item_features()

        embedding_table = self.loader.load_image_embeddings()

        interactions = self.loader.load_two_tower_interactions()

        interactions_df = interactions.to_pandas()

        return (
            user_table,
            item_table,
            embedding_table,
            interactions_df,
        )

    def build_artifacts(
        self,
        user_table,
        item_table,
        embedding_table,
    ):
        return self.lookup_builder.build(
            user_table=user_table,
            item_table=item_table,
            embedding_table=embedding_table,
        )

    def build_dataset(
        self,
        interactions_df,
        artifacts,
    ):
        dataset_builder = DatasetBuilder(artifacts)

        return dataset_builder.build_retrieval_dataset(
            interactions_df,
            batch_size=256,
        )

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
        bucket = "recommendation-system-1149"

        prefix = f"model_artifacts/{self.config.environment}/two_tower"

        s3 = boto3.client("s3")

        with tempfile.TemporaryDirectory() as tmp_dir:
            keras_path = Path(tmp_dir) / "two_tower_model.keras"

            model.save(keras_path)

            s3.upload_file(
                str(keras_path),
                bucket,
                f"{prefix}/two_tower_model.keras",
            )

            saved_model_path = Path(tmp_dir) / "saved_model"

            model.export(str(saved_model_path))

            for file in saved_model_path.rglob("*"):
                if file.is_file():
                    s3.upload_file(
                        str(file),
                        bucket,
                        f"{prefix}/saved_model/{file.relative_to(saved_model_path)}",
                    )

        print(f"Saved model to s3://{bucket}/{prefix}")

    def train(self):
        self.setup_device()

        (
            user_table,
            item_table,
            embedding_table,
            interactions_df,
        ) = self.load_data()

        artifacts = self.build_artifacts(
            user_table,
            item_table,
            embedding_table,
        )

        dataset = self.build_dataset(
            interactions_df,
            artifacts,
        )

        print("Dataset built successfully")

        model = self.build_model(artifacts)

        print("Model created")

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

        tracker = MLflowTracker(experiment_name="two_tower_retrieval")

        tracker.start_run(run_name=f"{self.config.environment}_run")

        params = {
            "environment": self.config.environment,
            "batch_size": 256,
            "epochs": 1,
            "learning_rate": 1e-3,
            "user_count": len(artifacts.customer_to_idx),
            "item_count": len(artifacts.article_to_idx),
        }

        tracker.log_params(params)

        history = model.fit(
            dataset,
            epochs=5,  # for test only 5
            verbose=1,
        )

        tracker.log_tensorflow_model(model)

        self.save_model_to_s3(model)

        metrics = {
            "final_loss": float(history.history["loss"][-1]),
            "min_loss": float(min(history.history["loss"])),
        }

        tracker.log_metrics(metrics)

        tracker.log_tensorflow_model(model)

        tracker.end_run()

        return history


def main():
    trainer = TwoTowerTrainer(
        environment="prod"  # set "test" for testing
    )

    trainer.train()


if __name__ == "__main__":
    main()
