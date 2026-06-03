from configs.test import TestConfig
from configs.prod import ProdConfig


def get_config(env: str):

    if env == "test":
        return TestConfig()

    if env == "prod":
        return ProdConfig()

    raise ValueError(
        f"Unknown env: {env}"
    )