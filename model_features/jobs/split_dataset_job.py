from common.constants import (
    RANKING_INTERACTIONS,
    TEST,
    TRAIN,
    TWO_TOWER_INTERACTIONS,
    VALIDATION,
)
from common.logger import get_logger
from configs.base import BaseConfig
from feature_engineering.splitter import DatasetSplitter
from storage.artifact_manager import ArtifactManager
from storage.s3_reader import S3Reader
from storage.s3_writer import S3Writer

logger = get_logger(__name__)


class SplitDatasetJob:
    def __init__(self, spark, config: BaseConfig):
        self.spark = spark
        self.config = config

        self.reader = S3Reader(spark, config)

        self.writer = S3Writer(config)

        self.artifacts = ArtifactManager(config)

    def run(self):
        logger.info("Starting SplitDatasetJob...")

        try:
            splitter = DatasetSplitter(self.config)

            # Ranking Dataset
            logger.info("Loading ranking features...")
            ranking_df = self.reader.load_ranking_features()
            logger.info(f"Loaded ranking data: {ranking_df.count()} rows")

            logger.info("Splitting ranking dataset...")
            (ranking_train, ranking_validation, ranking_test) = splitter.split(
                ranking_df
            )
            logger.info(
                f"Ranking split - Train: {ranking_train.count()}, Val: {ranking_validation.count()}, Test: {ranking_test.count()}"
            )

            logger.info("Writing ranking train data...")
            self.writer.write_parquet(ranking_train, f"{TRAIN}/{RANKING_INTERACTIONS}")

            logger.info("Writing ranking validation data...")
            self.writer.write_parquet(
                ranking_validation, f"{VALIDATION}/{RANKING_INTERACTIONS}"
            )

            logger.info("Writing ranking test data...")
            self.writer.write_parquet(ranking_test, f"{TEST}/{RANKING_INTERACTIONS}")

            # Two Tower Dataset
            logger.info("Loading two tower features...")
            two_tower_df = self.reader.load_two_tower_features()
            logger.info(f"Loaded two tower data: {two_tower_df.count()} rows")

            logger.info("Splitting two tower dataset...")
            (two_tower_train, two_tower_validation, two_tower_test) = splitter.split(
                two_tower_df
            )
            logger.info(
                f"Two tower split - Train: {two_tower_train.count()}, Val: {two_tower_validation.count()}, Test: {two_tower_test.count()}"
            )

            logger.info("Writing two tower train data...")
            self.writer.write_parquet(
                two_tower_train, f"{TRAIN}/{TWO_TOWER_INTERACTIONS}"
            )

            logger.info("Writing two tower validation data...")
            self.writer.write_parquet(
                two_tower_validation, f"{VALIDATION}/{TWO_TOWER_INTERACTIONS}"
            )

            logger.info("Writing two tower test data...")
            self.writer.write_parquet(
                two_tower_test, f"{TEST}/{TWO_TOWER_INTERACTIONS}"
            )

            logger.info("Building training metadata...")
            metadata = self.artifacts.build_training_metadata(
                ranking_train.count(), ranking_validation.count(), ranking_test.count()
            )

            logger.info("Saving split metadata...")
            self.artifacts.save_json(metadata, "split_metadata")

            logger.info("SplitDatasetJob completed successfully")

        except Exception as e:
            logger.error(f"SplitDatasetJob failed: {str(e)}", exc_info=True)
            raise
