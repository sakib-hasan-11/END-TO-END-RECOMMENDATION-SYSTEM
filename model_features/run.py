import argparse
import sys
import traceback
from datetime import datetime
from pyspark.sql import SparkSession

from configs.factory import get_config
from common.logger import setup_logger

from jobs.split_dataset_job import (
    SplitDatasetJob
)

from jobs.build_user_features_job import (
    BuildUserFeaturesJob
)

from jobs.build_item_features_job import (
    BuildItemFeaturesJob
)

from jobs.build_ranking_features_job import (
    BuildRankingFeaturesJob
)

# Setup logger
# logger = setup_logger(
#     "model_features"
# )
# os.makedirs(
#     "logs",
#     exist_ok=True
# )

def create_spark_session(logger):
    logger.info("Creating Spark session...")
    try:
        spark = (
            SparkSession.builder
            .appName(
                "recommendation-feature-pipeline"
            )
            .getOrCreate()
        )
        logger.info("Spark session created successfully")
        return spark
    except Exception as e:
        logger.error(f"Failed to create Spark session: {e}")
        raise


def main():
    
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        required=True,
        choices=["test", "prod"]
    )

    parser.add_argument(
        "--job",
        required=True,
        choices=[
            "split",
            "user",
            "item",
            "ranking",
            "all"
        ]
    )

    args = parser.parse_args()

    # Setup logger with environment-specific log file
    log_file = f"logs/model_features_{args.env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger("model_features", log_file)
    
    logger.info("=" * 80)
    logger.info(f"Starting Feature Pipeline - Environment: {args.env}, Job: {args.job}")
    logger.info("=" * 80)

    try:
        # Get config
        logger.info(f"Loading configuration for environment: {args.env}")
        config = get_config(args.env)
        config.validate()
        logger.info(f"Configuration validated - Output path: {config.output_root_path}")

        # Create Spark session
        spark = create_spark_session(logger)

        try:
            if args.job == "split":
                logger.info("Running SplitDatasetJob...")
                SplitDatasetJob(spark, config).run()
                logger.info("SplitDatasetJob completed successfully")

            elif args.job == "user":
                logger.info("Running BuildUserFeaturesJob...")
                BuildUserFeaturesJob(spark, config).run()
                logger.info("BuildUserFeaturesJob completed successfully")

            elif args.job == "item":
                logger.info("Running BuildItemFeaturesJob...")
                BuildItemFeaturesJob(spark, config).run()
                logger.info("BuildItemFeaturesJob completed successfully")

            elif args.job == "ranking":
                logger.info("Running BuildRankingFeaturesJob...")
                BuildRankingFeaturesJob(spark, config).run()
                logger.info("BuildRankingFeaturesJob completed successfully")

            elif args.job == "all":
                logger.info("Running full pipeline: split -> user -> item -> ranking")
                
                logger.info("Step 1/4: Running SplitDatasetJob...")
                SplitDatasetJob(spark, config).run()
                logger.info("Step 1/4: SplitDatasetJob completed")
                
                logger.info("Step 2/4: Running BuildUserFeaturesJob...")
                BuildUserFeaturesJob(spark, config).run()
                logger.info("Step 2/4: BuildUserFeaturesJob completed")
                
                logger.info("Step 3/4: Running BuildItemFeaturesJob...")
                BuildItemFeaturesJob(spark, config).run()
                logger.info("Step 3/4: BuildItemFeaturesJob completed")
                
                logger.info("Step 4/4: Running BuildRankingFeaturesJob...")
                BuildRankingFeaturesJob(spark, config).run()
                logger.info("Step 4/4: BuildRankingFeaturesJob completed")
                
                logger.info("All pipeline steps completed successfully")

        finally:
            logger.info("Stopping Spark session...")
            spark.stop()
            logger.info("Spark session stopped")
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error(
            f"Error traceback:\n{traceback.format_exc()}"
        )
        sys.exit(1)

    logger.info("=" * 80)
    logger.info(
        "Pipeline execution completed successfully"
    )
    logger.info("=" * 80)


if __name__ == "__main__":

    main()