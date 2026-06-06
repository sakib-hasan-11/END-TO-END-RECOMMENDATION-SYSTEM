from dataclasses import dataclass


@dataclass  # this method help to fetch data main paths from test/prod file when ever we run each of one. and then pipeline use this paths from this file
class BaseConfig:
    environment: str

    processed_root: str
    embeddings_root: str

    batch_size: int
    epochs: int
    MODEL_ARTIFACT_PATH: str

    @property # this helps to use the methods as attributes
    def user_features_path(self):
        return f"{self.processed_root}user_features/"


    @property
    def item_features_path(self):
        return f"{self.processed_root}item_features/"


    @property
    def two_tower_path(self):
        return f"{self.processed_root}two_tower_interactions/"


    @property
    def ranking_features_path(self):
        return f"{self.processed_root}ranking_features/"


    @property
    def ranking_interactions_path(self):
        return f"{self.processed_root}ranking_interactions/"
