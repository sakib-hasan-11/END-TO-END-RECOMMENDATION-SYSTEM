from common.constants import (
    ITEM_FEATURES,
    RANKING_FEATURES,
    RANKING_INTERACTIONS,
    TRAIN,
    USER_FEATURES,
)
from common.logger import get_logger
from feature_engineering.ranking_features import RankingFeatureBuilder
from storage.s3_reader import S3Reader
from storage.s3_writer import S3Writer
from monitoring.feature_statistics import (
    FeatureStatisticsBuilder
)

from storage.artifact_manager import (
    ArtifactManager
)


logger = get_logger(__name__)


class BuildRankingFeaturesJob:
    def __init__(self, spark, config):
        self.config = config

        self.artifact_manager=ArtifactManager(config=config)

        self.stat_builder=FeatureStatisticsBuilder()

        self.reader = S3Reader(spark, config)

        self.writer = S3Writer(config)

        self.builder = RankingFeatureBuilder(config)

    def run(self):
        logger.info("Starting BuildRankingFeaturesJob...")

        try:
            logger.info("Reading ranking interactions...")
            interactions_df = self.reader.read_parquet(
                f"{self.config.output_root_path}/{TRAIN}/{RANKING_INTERACTIONS}"
            )
            logger.info(f"Loaded interactions: {interactions_df.count()} rows")

            logger.info("Reading user features...")
            user_features = self.reader.read_parquet(
                f"{self.config.output_root_path}/{TRAIN}/{USER_FEATURES}"
            )
            logger.info(f"Loaded user features: {user_features.count()} rows")

            logger.info("Reading item features...")
            item_features = self.reader.read_parquet(
                f"{self.config.output_root_path}/{TRAIN}/{ITEM_FEATURES}"
            )
            logger.info(f"Loaded item features: {item_features.count()} rows")

            logger.info("Building ranking features...")
            ranking_features = self.builder.build(
                interactions_df, user_features, item_features
            )
            logger.info(
                f"Built ranking features: {ranking_features.count()} rows, {len(ranking_features.columns)} columns"
            )

            logger.info("Writing ranking features to S3...")
            self.writer.write_parquet(
                ranking_features,
                f"{TRAIN}/{RANKING_FEATURES}",
            )


            stats = self.stat_builder.build(
                ranking_features,
                "ranking_features"
            )

            self.artifact_manager.save_json(
                stats,
                "ranking_feature_statistics"
            )


            logger.info("BuildRankingFeaturesJob completed successfully")

        except Exception as e:
            logger.error(f"BuildRankingFeaturesJob failed: {str(e)}", exc_info=True)
            raise
