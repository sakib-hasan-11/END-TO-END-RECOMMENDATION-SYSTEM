# src/data/loaders.py

from common.s3_reader import S3Reader
from configs.base import BaseConfig


class DataLoader:

    def __init__(
        self,
        reader: S3Reader,
        config: BaseConfig,
    ):
        self.reader = reader
        self.config = config

    def load_user_features(self):
        return self.reader.read_dataset(
            self.config.user_features_path
        )

    def load_item_features(self):
        return self.reader.read_dataset(
            self.config.item_features_path
        )

    def load_image_embeddings(self):
        return self.reader.read_dataset(
            self.config.embeddings_root
        )

    def load_two_tower_interactions(self):
        return self.reader.read_dataset(
            self.config.two_tower_path
        )

    def load_ranking_features(self):
        return self.reader.read_dataset(
            self.config.ranking_features_path
        )

    def load_ranking_interactions(self):
        return self.reader.read_dataset(
            self.config.ranking_interactions_path
        )