import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pickle
import tempfile
from pathlib import Path

import boto3
import tensorflow as tf
from common.s3_reader import S3Reader
from configs import get_config
from data.loaders import DataLoader
from data.lookup_builder import LookupBuilder
from data.ranking_dataset import RankingDatasetBuilder
from models.ranking_model import RankingModel

from training.mlflow_tracker import MLflowTracker


class RankingTrainer:
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
        ranking_builder = RankingDatasetBuilder(artifacts)

        ranking_df = ranking_builder.build_training_dataframe(
            interactions_df,
            negative_ratio=3,
        )

        print(f"Training Rows: {len(ranking_df):,}")

        print(ranking_df["label"].value_counts(normalize=True))

        dataset = ranking_builder.build_tf_dataset(
            ranking_df,
            batch_size=256,
        )

        return dataset, ranking_df

    def build_model(self):
        return RankingModel()

    def save_lookup_artifacts_to_s3(
        self,
        artifacts,
    ):
        bucket = "recommendation-system-1149"

        prefix = f"model_artifacts/{self.config.environment}/lookups"

        s3 = boto3.client("s3")

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = Path(tmp_dir) / "lookup_artifacts.pkl"

            with open(
                artifact_path,
                "wb",
            ) as f:
                pickle.dump(
                    artifacts,
                    f,
                )

            s3.upload_file(
                str(artifact_path),
                bucket,
                f"{prefix}/lookup_artifacts.pkl",
            )

        print(f"Saved lookups to s3://{bucket}/{prefix}")

    def save_model_to_s3(
        self,
        model,
    ):
        bucket = "recommendation-system-1149"

        prefix = f"model_artifacts/{self.config.environment}/ranking"

        s3 = boto3.client("s3")

        with tempfile.TemporaryDirectory() as tmp_dir:
            keras_path = Path(tmp_dir) / "ranking_model.keras"

            model.save(keras_path)

            s3.upload_file(
                str(keras_path),
                bucket,
                f"{prefix}/ranking_model.keras",
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

        dataset, ranking_df = self.build_dataset(
            interactions_df,
            artifacts,
        )

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

        history = model.fit(
            dataset,
            epochs=5,
            verbose=1,
        )

        tracker.log_params(
            {
                "environment": self.config.environment,
                "batch_size": 256,
                "epochs": 5,
                "learning_rate": 1e-3,
            }
        )

        tracker.log_params(
            {
                "environment": self.config.environment,
                "batch_size": 256,
                "epochs": 5,
                "learning_rate": 1e-3,
                "training_rows": len(ranking_df),
                "user_count": len(artifacts.customer_to_idx),
                "item_count": len(artifacts.article_to_idx),
            }
        )

        tracker.log_tensorflow_model(
            model,
            artifact_path="ranking_model",
        )

        self.save_model_to_s3(model)

        self.save_lookup_artifacts_to_s3(artifacts)

        tracker.end_run()

        return model


def main():
    trainer = RankingTrainer(environment="prod")

    trainer.train()


if __name__ == "__main__":
    main()
