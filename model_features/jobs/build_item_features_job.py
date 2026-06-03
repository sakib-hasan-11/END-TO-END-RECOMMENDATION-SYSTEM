from common.constants import ITEM_FEATURES, TRAIN, TWO_TOWER_INTERACTIONS

from feature_engineering.item_features import ItemFeatureBuilder
from storage.s3_reader import S3Reader
from storage.s3_writer import S3Writer
from monitoring.feature_statistics import (
    FeatureStatisticsBuilder
)

from storage.artifact_manager import (
    ArtifactManager
)
from common.logger import get_logger
logger = get_logger(__name__)


class BuildItemFeaturesJob:
    def __init__(self, spark, config):

        self.config = config

        self.artifact_manager=ArtifactManager(config=config)

        self.stat_builder=FeatureStatisticsBuilder()

        self.reader = S3Reader(spark, config)

        self.writer = S3Writer(config)

        self.builder = ItemFeatureBuilder(config)

    def run(self):
        logger.info("Starting BuildItemFeaturesJob...")

        try:
            logger.info("Reading two tower interactions...")
            train_df = self.reader.read_parquet(
                f"{self.config.output_root_path}/{TRAIN}/{TWO_TOWER_INTERACTIONS}"
            )

            logger.info(f"Columns: {train_df.columns}")
            logger.info(f"Loaded two tower data: {train_df.count()} rows")

            logger.info("Loading embeddings...")
            embeddings_df = self.reader.load_embeddings()
            logger.info(f"Loaded embeddings: {embeddings_df.count()} rows")

            logger.info("Building item features...")
            item_features = self.builder.build(train_df, embeddings_df)
            logger.info(
                f"Built item features: {item_features.count()} rows, {len(item_features.columns)} columns"
            )

            stats = self.stat_builder.build(
                item_features,
                "item_features"
            )

            self.artifact_manager.save_json(
                stats,
                "item_feature_statistics"
            )

            logger.info("Writing item features to S3...")
            self.writer.write_parquet(
                item_features, f"{TRAIN}/{ITEM_FEATURES}"
            )
            logger.info("BuildItemFeaturesJob completed successfully")

        except Exception as e:
            logger.error(f"BuildItemFeaturesJob failed: {str(e)}", exc_info=True)
            raise
