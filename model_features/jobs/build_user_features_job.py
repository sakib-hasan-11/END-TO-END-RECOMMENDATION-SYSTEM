from common.constants import TRAIN, TWO_TOWER_INTERACTIONS, USER_FEATURES
from common.logger import get_logger
from feature_engineering.user_features import UserFeatureBuilder
from monitoring.feature_statistics import FeatureStatisticsBuilder
from storage.artifact_manager import ArtifactManager
from storage.s3_reader import S3Reader
from storage.s3_writer import S3Writer

logger = get_logger(__name__)


class BuildUserFeaturesJob:
    def __init__(self, spark, config):
        self.config = config

        self.artifact_manager = ArtifactManager(config=config)

        self.stat_builder = FeatureStatisticsBuilder()

        self.reader = S3Reader(spark, config)

        self.writer = S3Writer(config)

        self.builder = UserFeatureBuilder(config)

    def run(self):
        logger.info("Starting BuildUserFeaturesJob...")

        try:
            logger.info("Reading training data...")
            train_df = self.reader.read_parquet(
                f"{self.config.output_root_path}/{TRAIN}/{TWO_TOWER_INTERACTIONS}"
            )
            logger.info(f"Loaded training data: {train_df.count()} rows")
            logger.info(train_df.columns)

            logger.info("Building user features...")
            user_features = self.builder.build(train_df)
            logger.info(
                f"Built user features: {user_features.count()} rows, {len(user_features.columns)} columns"
            )

            stats = self.stat_builder.build(user_features, "user_features")

            self.artifact_manager.save_json(stats, "user_feature_statistics")

            logger.info("Writing user features to S3...")
            self.writer.write_parquet(
                user_features, f"{TRAIN}/{USER_FEATURES}"
            )


            logger.info("BuildUserFeaturesJob completed successfully")

        except Exception as e:
            logger.error(f"BuildUserFeaturesJob failed: {str(e)}", exc_info=True)
            raise
