from dataclasses import dataclass


@dataclass
class PipelineConfig:

    mode: str

    bucket: str

    input_zip_prefix: str

    output_embedding_prefix: str

    local_zip_dir: str

    local_image_dir: str

    local_output_dir: str

    batch_size: int


TEST_CONFIG = PipelineConfig(
    mode="test",
    bucket="recommendation-system-1149",
    input_zip_prefix="raw-data/sample_data/images/",
    output_embedding_prefix="features/sample/embeddings/",
    local_zip_dir="/tmp/zips",
    local_image_dir="/tmp/images",
    local_output_dir="/tmp/output",
    batch_size=100
)

PROD_CONFIG = PipelineConfig(
    mode="prod",
    bucket="recommendation-system-1149",
    input_zip_prefix="raw-data/main_data/images/",
    output_embedding_prefix="features/main/embeddings/",
    local_zip_dir="/tmp/zips",
    local_image_dir="/tmp/images",
    local_output_dir="/tmp/output",
    batch_size=1000
)