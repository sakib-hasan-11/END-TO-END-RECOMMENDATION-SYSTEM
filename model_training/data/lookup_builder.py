# src/data/lookup_builder.py

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


@dataclass
class LookupArtifacts:
    customer_to_idx: Dict[str, int]
    article_to_idx: Dict[str, int]

    user_feature_matrix: np.ndarray

    item_numeric_matrix: np.ndarray
    item_categorical_matrix: np.ndarray

    image_embedding_matrix: np.ndarray

    article_embedding_lookup: Dict[str, np.ndarray]

    favorite_category_vocab: List[str]
    favorite_color_vocab: List[str]


class LookupBuilder:
    def align_item_tables(
        self,
        item_table: pa.Table,
        embedding_table: pa.Table,
    ):
        """
        Keep only items that exist in both
        item features and image embeddings.

        Returns:
            aligned_item_table,
            aligned_embedding_table
        """

        item_ids = set(item_table.column("article_id").to_pylist())

        embedding_ids = set(embedding_table.column("article_id").to_pylist())

        common_ids = item_ids.intersection(embedding_ids)

        print(f"Aligned items: {len(common_ids):,}")

        item_mask = pc.is_in(
            item_table["article_id"], value_set=pa.array(list(common_ids))
        )

        embedding_mask = pc.is_in(
            embedding_table["article_id"], value_set=pa.array(list(common_ids))
        )

        aligned_item_table = item_table.filter(item_mask)

        aligned_embedding_table = embedding_table.filter(embedding_mask)

        # IMPORTANT:
        # Ensure identical ordering

        aligned_item_table = aligned_item_table.sort_by("article_id")

        aligned_embedding_table = aligned_embedding_table.sort_by("article_id")

        def build_embedding_lookup(
            self,
            embedding_table: pa.Table,
        ) -> Dict[str, np.ndarray]:
            article_ids = embedding_table.column("article_id").to_pylist()

            embeddings = embedding_table.column("image_embedding").to_pylist()

            return {
                article_id: np.asarray(
                    embedding,
                    dtype=np.float32,
                )
                for article_id, embedding in zip(
                    article_ids,
                    embeddings,
                )
            }

        return (
            aligned_item_table,
            aligned_embedding_table,
        )

    def build_user_lookup(
        self,
        user_table: pa.Table,
    ) -> Dict[str, int]:
        customer_ids = user_table.column("customer_id").to_pylist()

        return {customer_id: idx for idx, customer_id in enumerate(customer_ids)}

    def build_item_lookup(
        self,
        item_table: pa.Table,
    ) -> Dict[str, int]:
        article_ids = item_table.column("article_id").to_pylist()

        return {article_id: idx for idx, article_id in enumerate(article_ids)}

    def build_user_feature_matrix(
        self,
        user_table: pa.Table,
    ) -> np.ndarray:
        features = [
            "purchase_count",
            "avg_spend",
            "weekend_ratio",
            "category_diversity",
            "preferred_channel",
            "avg_purchase_gap_days",
        ]

        columns = []

        for feature in features:
            column = (
                user_table.column(feature).fill_null(0).to_numpy().astype(np.float32)
            )

            columns.append(column)

        return np.column_stack(columns)

    def build_item_numeric_matrix(
        self,
        item_table: pa.Table,
    ) -> np.ndarray:
        features = [
            "item_purchase_count",
            "unique_buyers",
            "avg_item_price",
            "days_since_last_sale",
        ]

        columns = []

        for feature in features:
            column = (
                item_table.column(feature).fill_null(0).to_numpy().astype(np.float32)
            )

            columns.append(column)

        return np.column_stack(columns)

    def build_item_categorical_matrix(
        self,
        item_table: pa.Table,
    ) -> np.ndarray:
        features = [
            "department_no",
            "product_type_no",
            "section_no",
            "garment_group_no",
            "colour_group_code",
            "index_group_no",
        ]

        columns = []

        for feature in features:
            column = (
                item_table.column(feature).fill_null(-1).to_numpy().astype(np.int32)
            )

            columns.append(column)

        return np.column_stack(columns)

    def build_embedding_matrix(
        self,
        embedding_table: pa.Table,
    ) -> np.ndarray:
        embeddings = embedding_table.column("image_embedding").to_pylist()

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def build_user_category_vocab(
        self,
        user_table: pa.Table,
    ) -> List[str]:
        values = user_table.column("favorite_category").drop_null().to_pylist()

        return sorted(list(set(values)))

    def build_user_color_vocab(
        self,
        user_table: pa.Table,
    ) -> List[str]:
        values = user_table.column("favorite_color").drop_null().to_pylist()

        return sorted(list(set(values)))

    def build_embedding_lookup(
        self,
        embedding_table: pa.Table,
    ) -> Dict[str, np.ndarray]:
        article_ids = embedding_table.column("article_id").to_pylist()

        embeddings = embedding_table.column("image_embedding").to_pylist()

        return {
            article_id: np.asarray(
                embedding,
                dtype=np.float32,
            )
            for article_id, embedding in zip(
                article_ids,
                embeddings,
            )
        }

    def build(
        self,
        user_table: pa.Table,
        item_table: pa.Table,
        embedding_table: pa.Table,
    ) -> LookupArtifacts:
        item_table, embedding_table = self.align_item_tables(
            item_table,
            embedding_table,
        )

        return LookupArtifacts(
            customer_to_idx=self.build_user_lookup(user_table),
            article_to_idx=self.build_item_lookup(item_table),
            user_feature_matrix=self.build_user_feature_matrix(user_table),
            item_numeric_matrix=self.build_item_numeric_matrix(item_table),
            item_categorical_matrix=self.build_item_categorical_matrix(item_table),
            image_embedding_matrix=self.build_embedding_matrix(embedding_table),
            article_embedding_lookup=self.build_embedding_lookup(embedding_table),
            favorite_category_vocab=self.build_user_category_vocab(user_table),
            favorite_color_vocab=self.build_user_color_vocab(user_table),
        )
