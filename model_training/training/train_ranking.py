import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

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

        dataset = ranking_builder.build_tf_dataset(
            ranking_df,
            batch_size=256,
        )

        return dataset

    def build_model(self):
        return RankingModel()

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
            epochs=1,
            verbose=1,
        )

        tracker.log_params(
            {
                "environment": self.config.environment,
                "batch_size": 256,
                "epochs": 1,
                "learning_rate": 1e-3,
            }
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

        tracker.end_run()

        return model


def main():

    trainer = RankingTrainer(
        environment="test"
    )

    trainer.train()


if __name__ == "__main__":
    main()