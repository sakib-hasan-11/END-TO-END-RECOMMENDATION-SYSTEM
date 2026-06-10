from typing import List, Optional

import pyarrow.dataset as ds

# import s3fs
import pyarrow.fs as pafs

# import pandas as pd
# import pyarrow as pa
# import pyarrow.dataset as ds
import pyarrow.parquet as pq


class S3Reader:
    """
    Generic S3 parquet reader.

    Supports:
    - Single parquet files
    - Partitioned parquet datasets
    - Column projection
    - PyArrow output
    - Pandas output
    """

    def __init__(self):
        # self.fs = s3fs.S3FileSystem()

        self.fs = pafs.S3FileSystem(region="us-east-1")

    def read_parquet(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        to_pandas: bool = False,
    ):
        """
        Read a single parquet file.

        Args:
            path: S3 parquet file path
            columns: Optional selected columns
            to_pandas: Return pandas dataframe if True

        Returns:
            pyarrow.Table or pandas.DataFrame
        """

        table = pq.read_table(
            path,
            filesystem=self.fs,
            columns=columns,
        )

        if to_pandas:
            return table.to_pandas()

        return table

    def _normalize_path(self, path: str) -> str:
        if path.startswith("s3://"):
            return path.replace("s3://", "", 1)
        return path

    def read_dataset(
        self,
        path: str,
        columns=None,
        to_pandas: bool = False,
    ):
        path = self._normalize_path(path)

        table = pq.read_table(
            path,
            filesystem=self.fs,
            columns=columns,
        )

        if to_pandas:
            return table.to_pandas()

        return table

    def read_dataset_streaming(
        self,
        path: str,
        columns=None,
    ):
        path = self._normalize_path(path)

        dataset = ds.dataset(
            path,
            filesystem=self.fs,
            format="parquet",
        )

        return dataset
