import sys

from config.config import PROD_CONFIG, TEST_CONFIG
from src.logger import get_logger
from src.pipeline import EmbeddingPipeline

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Image Embedding Pipeline - Starting Execution")
    logger.info("=" * 80)

    try:
        # Determine mode from command line or default to TEST
        mode = sys.argv[1] if len(sys.argv) > 1 else "test"

        if mode.lower() == "prod":
            logger.info(f"Running in PRODUCTION mode")
            config = PROD_CONFIG
        else:
            logger.info(f"Running in TEST mode")
            config = TEST_CONFIG

        pipeline = EmbeddingPipeline(config)

        pipeline.run()

        logger.info("=" * 80)
        logger.info("Pipeline execution completed successfully!")
        logger.info("=" * 80)
        sys.exit(0)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Pipeline execution failed: {str(e)}")
        logger.error("=" * 80)
        sys.exit(1)
