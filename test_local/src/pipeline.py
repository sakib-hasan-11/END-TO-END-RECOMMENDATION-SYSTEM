import shutil
import time
from datetime import datetime
from pathlib import Path

from src.data_loader import S3DataLoader
from src.embedding_generator import CLIPEmbeddingGenerator
from src.embedding_writer import EmbeddingWriter
from src.image_extractor import ImageExtractor
from src.logger import get_logger
from src.s3_writer import S3Writer

logger = get_logger(__name__)


class EmbeddingPipeline:
    def __init__(self, config):
        logger.info(f"Initializing EmbeddingPipeline in mode: {config.mode}")
        self.config = config

        try:
            logger.info("Initializing pipeline components")
            self.loader = S3DataLoader(config)
            logger.info("✓ S3DataLoader initialized")

            self.extractor = ImageExtractor(config)
            logger.info("✓ ImageExtractor initialized")

            self.embedder = CLIPEmbeddingGenerator()
            logger.info("✓ CLIPEmbeddingGenerator initialized")

            self.writer = EmbeddingWriter(config)
            logger.info("✓ EmbeddingWriter initialized")

            self.s3_writer = S3Writer(config)
            logger.info("✓ S3Writer initialized")
            logger.info("All pipeline components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {str(e)}")
            raise

    def run(self):
        logger.info("Starting embedding pipeline execution")
        pipeline_start_time = time.time()

        try:
            zip_files = self.loader.list_zip_files()
            logger.info(f"Zip files to process: {zip_files}")

            for zip_id, zip_key in enumerate(zip_files):
                logger.info(
                    f"\n--- Processing zip {zip_id + 1}/{len(zip_files)}: {zip_key} ---"
                )
                zip_start_time = time.time()

                try:
                    logger.info("Downloading zip files...")
                    zip_path = self.loader.download_zip(zip_key)

                    logger.info("Extracting zip files...")
                    image_dir = self.extractor.extract_zip(zip_path)

                    logger.info("Packing the images...")
                    images = list(Path(image_dir).glob("*.jpg"))
                    logger.info(f"Found {len(images)} jpg images")

                    images = images[:1000]
                    total_batches = (
                        len(images) + self.config.batch_size - 1
                    ) // self.config.batch_size
                    logger.info(
                        f"Processing {len(images)} images in {total_batches} batches (batch_size: {self.config.batch_size})"
                    )

                    for batch_id, batch_start in enumerate(
                        range(0, len(images), self.config.batch_size)
                    ):
                        batch_time = time.time()
                        batch = images[
                            batch_start : batch_start + self.config.batch_size
                        ]

                        logger.info(
                            f"Processing batch {batch_id + 1}/{total_batches} with {len(batch)} images"
                        )

                        records = []
                        failed_images = 0

                        # Process batch in GPU batches
                        total_gpu_batches = (
                            len(batch) + self.config.gpu_image_batch - 1
                        ) // self.config.gpu_image_batch

                        for gpu_batch_id, gpu_batch_start in enumerate(
                            range(0, len(batch), self.config.gpu_image_batch)
                        ):
                            gpu_batch = batch[
                                gpu_batch_start : gpu_batch_start
                                + self.config.gpu_image_batch
                            ]

                            logger.debug(
                                f"Processing GPU batch {gpu_batch_id + 1}/{total_gpu_batches} with {len(gpu_batch)} images"
                            )

                            try:
                                embeddings, valid_paths = (
                                    self.embedder.generate_batch_embeddings(
                                        [str(img_path) for img_path in gpu_batch]
                                    )
                                )

                                for img_path, embedding in zip(valid_paths, embeddings):
                                    article_id = Path(img_path).stem
                                    records.append(
                                        {
                                            "article_id": article_id,
                                            "image_embedding": embedding,
                                            "event_timestamp": datetime.utcnow(),
                                        }
                                    )

                                failed_images += len(gpu_batch) - len(valid_paths)

                            except Exception as e:
                                logger.warning(
                                    f"Failed to process GPU batch {gpu_batch_id + 1}: {str(e)}"
                                )
                                failed_images += len(gpu_batch)
                                continue

                        logger.info(
                            f"Batch {batch_id + 1} processed: {len(records)} successful, {failed_images} failed"
                        )

                        if len(records) == 0:
                            logger.warning(
                                f"Skipping batch {batch_id + 1} - no valid embeddings generated"
                            )
                            continue

                        parquet_file = (
                            f"{self.config.local_output_dir}/"
                            f"zip_{zip_id}"
                            f"_batch_{batch_start}.parquet"
                        )
                        logger.info("Writing to parquet format...")
                        self.writer.write_parquet(records, parquet_file)

                        s3_key = (
                            f"{self.config.output_embedding_prefix}"
                            f"zip_{zip_id}"
                            f"_batch_{batch_start}.parquet"
                        )
                        logger.info("Saving to S3...")
                        self.s3_writer.upload_file(parquet_file, s3_key)

                        batch_duration = time.time() - batch_time
                        logger.info(
                            f"Batch {batch_id + 1} completed in {batch_duration:.2f}s"
                        )

                    logger.info(f"Cleaning up local image directory")
                    shutil.rmtree(self.config.local_image_dir, ignore_errors=True)

                    zip_duration = time.time() - zip_start_time
                    logger.info(f"Zip {zip_id + 1} completed in {zip_duration:.2f}s")
                except Exception as e:
                    logger.error(f"Error processing zip {zip_id}: {str(e)}")
                    logger.info(f"Cleaning up failed batch...")
                    shutil.rmtree(self.config.local_image_dir, ignore_errors=True)
                    raise

            total_duration = time.time() - pipeline_start_time
            logger.info(
                f"\n*** Pipeline execution completed successfully in {total_duration:.2f}s ***"
            )
        except Exception as e:
            total_duration = time.time() - pipeline_start_time
            logger.error(
                f"Pipeline execution failed after {total_duration:.2f}s: {str(e)}"
            )
            raise
