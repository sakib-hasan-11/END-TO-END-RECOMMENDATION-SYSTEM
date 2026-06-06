import tensorflow as tf
import tensorflow_recommenders as tfrs


class UserTower(tf.keras.Model):
    def __init__(
        self,
        category_vocab,
        color_vocab,
        embedding_dim=32,
    ):
        super().__init__()

        self.category_lookup = tf.keras.layers.StringLookup(
            vocabulary=category_vocab,
            mask_token=None,
        )

        self.color_lookup = tf.keras.layers.StringLookup(
            vocabulary=color_vocab,
            mask_token=None,
        )

        self.category_embedding = tf.keras.layers.Embedding(
            input_dim=len(category_vocab) + 1,
            output_dim=embedding_dim,
        )

        self.color_embedding = tf.keras.layers.Embedding(
            input_dim=len(color_vocab) + 1,
            output_dim=embedding_dim,
        )

        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64),
            ]
        )

    def call(self, inputs):
        numeric = inputs["user_numeric"]

        category = self.category_embedding(
            self.category_lookup(inputs["user_category"])
        )

        color = self.color_embedding(self.color_lookup(inputs["user_color"]))

        x = tf.concat(
            [
                numeric,
                category,
                color,
            ],
            axis=1,
        )

        return self.dense(x)


class ItemTower(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.dense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    256,
                    activation="relu",
                ),
                tf.keras.layers.Dense(64),
            ]
        )

    def call(self, inputs):
        x = tf.concat(
            [
                inputs["item_numeric"],
                inputs["image_embedding"],
            ],
            axis=1,
        )

        return self.dense(x)


class TwoTowerModel(tfrs.models.Model):
    def __init__(
        self,
        category_vocab,
        color_vocab,
    ):
        super().__init__()

        self.user_model = UserTower(
            category_vocab=category_vocab,
            color_vocab=color_vocab,
        )

        self.item_model = ItemTower()

        self.task = tfrs.tasks.Retrieval()

    def compute_loss(
        self,
        features,
        training=False,
    ):
        user_embeddings = self.user_model(features)

        item_embeddings = self.item_model(features)

        return self.task(
            user_embeddings,
            item_embeddings,
        )
