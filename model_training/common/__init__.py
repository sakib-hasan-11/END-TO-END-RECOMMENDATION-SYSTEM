from configs.test import TestConfig
from configs.prod import ProdConfig


def get_config(environment: str):

    if environment == "test":
        return TestConfig()

    if environment == "prod":
        return ProdConfig()

    raise ValueError(f"Unknown environment: {environment}")