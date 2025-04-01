"""Configuration modules."""

from dotenv.main import dotenv_values
from pydantic import BaseModel

from libs.utils import common

# ======================================================================================


class Configs(BaseModel):
    """Model for Application configs."""


# ======================================================================================


def _load_env(env_file: str = ".env") -> dict[str, str | None]:
    """Load credentials from .env file.

    Args:
    ----
    env_file (str) : path to .env file

    """
    env_data = dotenv_values(env_file)

    return env_data


def load_configs(yaml_file: str, env_file: str = ".env") -> Configs:
    """Load configurations from configs file and environment.

    Args:
    ----
    yaml_file (str) : configuration file in YAML format
    env_file (str) : environment file
        (default: .env)

    Returns:
    (Configs) : loaded configurations

    """
    cfgs = Configs(**common.load_yaml(yaml_file))

    return cfgs
