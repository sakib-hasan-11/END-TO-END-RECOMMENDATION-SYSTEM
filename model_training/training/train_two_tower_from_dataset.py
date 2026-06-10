import pickle
import tempfile
from pathlib import Path

import boto3
import tensorflow as tf
from configs import get_config
from models.two_tower import TwoTowerModel

from training.mlflow_tracker import MLflowTracker


class TwoTowerTrainer:
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

        prefix = f"prepared_datasets/{self.config.environment}/two_tower"

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

        artifact_path = local_dir / "lookup_artifacts.pkl"

        s3.download_file(
            bucket,
            f"{prefix}/lookup_artifacts.pkl",
            str(artifact_path),
        )

        with open(
            artifact_path,
            "rb",
        ) as f:
            artifacts = pickle.load(f)

        return dataset, artifacts

    def build_model(
        self,
        artifacts,
    ):
        return TwoTowerModel(
            category_vocab=artifacts.favorite_category_vocab,
            color_vocab=artifacts.favorite_color_vocab,
        )

    def train(self):
        self.setup_device()

        dataset, artifacts = self.load_dataset()

        model = self.build_model(artifacts)

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

        tracker = MLflowTracker(
            experiment_name="two_tower_retrieval"
        )  # mlflow experiment name .

        tracker.start_run(run_name="prod_run")

        history = model.fit(
            dataset,
            epochs=5,
        )

        tracker.log_metrics({"final_loss": float(history.history["loss"][-1])})

        tracker.log_tensorflow_model(model)

        tracker.end_run()


def main():
    trainer = TwoTowerTrainer(environment="prod")

    trainer.train()


if __name__ == "__main__":
    main()


# python -m training.train_two_tower_from_dataset
