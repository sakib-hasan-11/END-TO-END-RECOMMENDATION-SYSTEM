import os
import pickle
import tempfile
from pathlib import Path

import boto3

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from configs import get_config
from models.ranking_model import RankingModel

from training.mlflow_tracker import MLflowTracker


class RankingTrainer:
    def __init__(
        self,
        environment="prod",
    ):
        self.config = get_config(environment)

    def setup_device(self):
        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(
                    gpu,
                    True,
                )

            print(f"Using GPU: {len(gpus)}")

    def load_dataset(self):
        bucket = "recommendation-system-1149"

        prefix = f"prepared_datasets/{self.config.environment}/ranking"

        s3 = boto3.client("s3")

        temp_dir = tempfile.TemporaryDirectory()

        local_dir = Path(temp_dir.name)

        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=f"{prefix}/dataset/",
        ):
            for obj in page.get(
                "Contents",
                [],
            ):
                key = obj["Key"]

                relative = key.replace(
                    f"{prefix}/dataset/",
                    "",
                )

                target = local_dir / relative

                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                s3.download_file(
                    bucket,
                    key,
                    str(target),
                )

        dataset = tf.data.Dataset.load(str(local_dir))

        artifact_file = local_dir / "lookup_artifacts.pkl"

        s3.download_file(
            bucket,
            f"{prefix}/lookup_artifacts.pkl",
            str(artifact_file),
        )

        with open(
            artifact_file,
            "rb",
        ) as f:
            artifacts = pickle.load(f)

        return dataset, artifacts

    def build_model(self):
        return RankingModel()

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

            saved_model = Path(tmp_dir) / "saved_model"

            model.export(str(saved_model))

            for file in saved_model.rglob("*"):
                if file.is_file():
                    s3.upload_file(
                        str(file),
                        bucket,
                        f"{prefix}/saved_model/{file.relative_to(saved_model)}",
                    )

    def train(self):
        self.setup_device()

        dataset, artifacts = self.load_dataset()

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

        tracker.log_metrics(
            {
                "final_loss": float(history.history["loss"][-1]),
                "final_auc": float(history.history["auc"][-1]),
                "final_accuracy": float(history.history["accuracy"][-1]),
            }
        )

        tracker.log_tensorflow_model(
            model,
            artifact_path="ranking_model",
        )

        self.save_model_to_s3(model)

        tracker.end_run()


def main():
    trainer = RankingTrainer(environment="prod")

    trainer.train()


if __name__ == "__main__":
    main()


# python -m training.train_two_tower_from_dataset