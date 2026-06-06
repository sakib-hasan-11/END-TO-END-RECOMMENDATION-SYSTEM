from common.s3_reader import S3Reader
from configs import get_config
from data.loaders import DataLoader
from data.lookup_builder import LookupBuilder
from data.ranking_dataset import RankingDatasetBuilder

from models.ranking_model import RankingModel


config = get_config("test")

reader = S3Reader()

loader = DataLoader(
    reader=reader,
    config=config,
)

builder = LookupBuilder()

user_table = loader.load_user_features()
item_table = loader.load_item_features()
embedding_table = loader.load_image_embeddings()
interactions = loader.load_two_tower_interactions()

interactions_df = interactions.to_pandas()
artifacts = builder.build(
    user_table=user_table,
    item_table=item_table,
    embedding_table=embedding_table,
)

builder = RankingDatasetBuilder(artifacts)

ranking_df = builder.build_training_dataframe(
    interactions_df,
    negative_ratio=3,
)

print(ranking_df["label"].value_counts())
print(ranking_df.shape)

print(ranking_df["label"].value_counts(normalize=True))
ranking_builder = RankingDatasetBuilder(artifacts)
dataset = ranking_builder.build_tf_dataset(
    ranking_df,
    batch_size=256,
)

for features, labels in dataset.take(1):
    print(features["user_numeric"].shape)

    print(features["item_numeric"].shape)

    print(features["image_embedding"].shape)

    print(labels.shape)



model = RankingModel()

for features, labels in dataset.take(1):

    predictions = model(
        features
    )

    print(
        predictions.shape
    )

    print(
        predictions[:5]
    )