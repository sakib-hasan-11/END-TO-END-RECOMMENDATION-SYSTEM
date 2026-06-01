import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Set log level from environment variable
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "./logs")
LOG_FILE = os.path.join(LOG_DIR, "embedding_pipeline.log")


def get_logger(name):
    """
    Get a logger instance with console and file handlers.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, LOG_LEVEL))

        # File handler with rotation
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setLevel(getattr(logging, LOG_LEVEL))

        # Formatter with more detail
        detailed_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Simple formatter for console
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(detailed_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
