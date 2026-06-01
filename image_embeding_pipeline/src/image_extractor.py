import zipfile
from pathlib import Path

from src.logger import get_logger

logger = get_logger(__name__)


class ImageExtractor:
    def __init__(self, config):
        logger.info(
            f"Initializing ImageExtractor with local_image_dir: {config.local_image_dir}"
        )
        self.config = config

    def extract_zip(self, zip_path):
        logger.info(f"Extracting zip file: {zip_path}")
        try:
            image_dir = Path(self.config.local_image_dir)

            image_dir.mkdir(parents=True, exist_ok=True)

            extracted_count = 0

            with zipfile.ZipFile(zip_path, "r") as z:
                logger.info(f"Zip file contains {len(z.namelist())} total members")

                for member in z.namelist():
                    if member.endswith(".jpg"):
                        # Path traversal check
                        if ".." in member or member.startswith("/"):
                            logger.warning(f"Skipping suspicious path: {member}")
                            continue

                        filename = Path(member).name

                        target = image_dir / filename

                        with z.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                        extracted_count += 1

            logger.info(
                f"Successfully extracted {extracted_count} jpg images to {image_dir}"
            )
            return image_dir
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file {zip_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to extract zip file {zip_path}: {str(e)}")
            raise
