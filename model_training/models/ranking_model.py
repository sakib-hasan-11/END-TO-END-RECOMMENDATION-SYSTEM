import tensorflow as tf


class RankingModel(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.network = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    256,
                    activation="relu",
                ),

                tf.keras.layers.Dropout(
                    0.2
                ),

                tf.keras.layers.Dense(
                    128,
                    activation="relu",
                ),

                tf.keras.layers.Dropout(
                    0.2
                ),

                tf.keras.layers.Dense(
                    64,
                    activation="relu",
                ),

                tf.keras.layers.Dense(
                    1,
                    activation="sigmoid",
                ),
            ]
        )

    def call(
        self,
        features,
    ):

        x = tf.concat(
            [
                features["user_numeric"],
                features["item_numeric"],
                features["image_embedding"],
            ],
            axis=1,
        )

        return self.network(x)