from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)


class FeatureStatisticsBuilder:
    def __init__(self):
        pass

    def build(self, df: DataFrame, dataset_name: str):

        row_count = df.count()

        numeric_cols = [
            field.name
            for field in df.schema.fields
            if isinstance(
                field.dataType,
                (
                    IntegerType,
                    LongType,
                    FloatType,
                    DoubleType,
                    DecimalType,
                    ShortType,
                ),
            )
        ]

        categorical_cols = [
            field.name
            for field in df.schema.fields
            if field.dataType.simpleString() == "string"
        ]

        embedding_cols = [
            field.name
            for field in df.schema.fields
            if isinstance(field.dataType, ArrayType)
        ]

        result = {
            "dataset_name": dataset_name,
            "row_count": row_count,
            "column_count": len(df.columns),

            # Full schema information
            "columns": {
                field.name: field.dataType.simpleString()
                for field in df.schema.fields
            },

            "numeric_features": {},
            "categorical_features": {},
            "embedding_features": {},
        }

        #
        # Numeric Features
        #

        for col_name in numeric_cols:

            stats_row = df.agg(
                F.count(col_name).alias("count"),
                F.count(
                    F.when(
                        F.col(col_name).isNull(),
                        1
                    )
                ).alias("null_count"),
                F.min(col_name).alias("min"),
                F.max(col_name).alias("max"),
                F.mean(col_name).alias("mean"),
                F.stddev(col_name).alias("std"),
            ).collect()[0]

            percentiles = (
                df.select(
                    F.percentile_approx(
                        col_name,
                        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
                    ).alias("percentiles")
                )
                .collect()[0]["percentiles"]
            )

            # Handle all-null columns
            if percentiles is None:
                percentiles = [None] * 7

            result["numeric_features"][col_name] = {
                "dtype": df.schema[col_name].dataType.simpleString(),
                "count": int(stats_row["count"]),
                "null_count": int(stats_row["null_count"]),
                "min": stats_row["min"],
                "max": stats_row["max"],
                "mean": stats_row["mean"],
                "std": stats_row["std"],
                "p01": percentiles[0],
                "p05": percentiles[1],
                "p25": percentiles[2],
                "p50": percentiles[3],
                "p75": percentiles[4],
                "p95": percentiles[5],
                "p99": percentiles[6],
            }

        #
        # Categorical Features
        #

        for col_name in categorical_cols:

            unique_count = (
                df.select(col_name)
                .distinct()
                .count()
            )

            top_values = (
                df.groupBy(col_name)
                .count()
                .orderBy(F.desc("count"))
                .limit(10)
                .collect()
            )

            result["categorical_features"][col_name] = {
                "dtype": "string",
                "unique_count": unique_count,
                "top_values": {
                    str(row[col_name]): int(row["count"])
                    for row in top_values
                },
            }

        #
        # Embedding Features
        #

        for col_name in embedding_cols:

            stats = (
                df.select(
                    F.size(col_name).alias("embedding_dim")
                )
                .agg(
                    F.count("*").alias("count"),
                    F.avg("embedding_dim").alias(
                        "avg_dimension"
                    ),
                )
                .collect()[0]
            )

            result["embedding_features"][col_name] = {
                "dtype": df.schema[col_name]
                .dataType
                .simpleString(),
                "count": int(stats["count"]),
                "avg_dimension": (
                    float(stats["avg_dimension"])
                    if stats["avg_dimension"] is not None
                    else None
                ),
            }

        #
        # Embedding Coverage
        #

        if "embedding_available" in df.columns:

            embedding_stats = (
                df.agg(
                    F.mean(
                        "embedding_available"
                    ).alias("coverage_ratio")
                )
                .collect()[0]
            )

            result["embedding_statistics"] = {
                "coverage_ratio": embedding_stats[
                    "coverage_ratio"
                ]
            }

        return result