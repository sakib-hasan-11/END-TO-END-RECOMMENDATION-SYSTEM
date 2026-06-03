from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class UserFeatureSchema:

    ID_COLUMN = "customer_id"

    NUMERIC_FEATURES = [
        "purchase_count",
        "avg_spend",
        "weekend_ratio",
        "category_diversity",
        "avg_purchase_gap_days",
    ]

    CATEGORICAL_FEATURES = [
        "preferred_channel",
        "favorite_category",
        "favorite_color",
    ]

    ALL_FEATURES = (
        NUMERIC_FEATURES +
        CATEGORICAL_FEATURES
    )


@dataclass(frozen=True)
class ItemFeatureSchema:

    ID_COLUMN = "article_id"

    NUMERIC_FEATURES = [
        "item_purchase_count",
        "unique_buyers",
        "avg_item_price",
    ]

    CATEGORICAL_FEATURES = [
        "department_no",
        "product_type_no",
        "section_no",
        "garment_group_no",
        "colour_group_code",
        "index_group_no",
    ]

    IMAGE_FEATURE_COLUMN = "image_embedding"

    ALL_FEATURES = (
        NUMERIC_FEATURES +
        CATEGORICAL_FEATURES +
        [IMAGE_FEATURE_COLUMN]
    )


@dataclass(frozen=True)
class RankingFeatureSchema:

    USER_ID_COLUMN = "customer_id"

    ITEM_ID_COLUMN = "article_id"

    LABEL_COLUMN = "label"

    NUMERIC_FEATURES = [

        "user_avg_price",

        "user_total_interactions",

        "user_category_diversity",

        "item_popularity",

        "item_avg_price",

        "unique_buyers",

        "price_affinity",

        "popularity_price_score",

        "days_since_last_purchase",

        "day_of_week",

        "month",

        "is_weekend"
    ]

    ALL_FEATURES = (
        NUMERIC_FEATURES +
        [LABEL_COLUMN]
    )