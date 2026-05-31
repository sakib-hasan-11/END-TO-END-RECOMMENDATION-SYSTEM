from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("image_embedding", pa.list_(pa.float32())),
        ("event_timestamp", pa.timestamp("us")),
    ]
)


class EmbeddingWriter:
    def __init__(self, config):
        logger.info(
            f"Initializing EmbeddingWriter with output_dir: {config.local_output_dir}"
        )
        self.config = config

    def write_parquet(self, records, output_file):
        logger.info(f"Writing {len(records)} records to parquet: {output_file}")
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directory created")

            table = pa.Table.from_pylist(records, schema=EMBEDDING_SCHEMA)
            logger.debug(f"PyArrow table created with {len(records)} rows")

            pq.write_table(table, output_file, compression="snappy")

            file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Successfully wrote parquet file (Size: {file_size:.2f} MB)")
            return output_file
        except Exception as e:
            logger.error(f"Failed to write parquet file {output_file}: {str(e)}")
            raise
