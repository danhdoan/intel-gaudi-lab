"""Configuration modules."""

from typing import Optional, Dict

from pydantic import BaseModel
from dotenv.main import dotenv_values

from enouvo.utils import common


# ======================================================================================


class Dataset(BaseModel):
    """Model for Dataset config."""

    dataset_path: str


class TrainingSettings(BaseModel):
    """Model for Training procedure."""

    lr: float
    num_epochs: int
    batch_size: int


class Credentials(BaseModel):
    """Model for Credentials."""

    api_key: Optional[str] = None
    salt_key: Optional[str] = None


class Configs(BaseModel):
    """Model for Application configs."""

    dataset: Dataset
    training: TrainingSettings
    credentials: Optional[Credentials] = None


# ======================================================================================


def _load_env(env_file: str = '.env') -> Dict[str, Optional[str]]:
    """Load credentials from .env file.

    Args:
    ----
    env_file (str) : path to .env file
    """
    env_data = dotenv_values(env_file)

    return env_data



def load_configs(yaml_file: str, env_file: str = '.env') -> Configs:
    """Load configurations from configs file and environment.

    Args:
    ----
    yaml_file (str) : configuration file in YAML format

    Returns:
    (Configs) : loaded configurations
    """

    cfgs = Configs(**common.load_yaml(yaml_file))
    cfgs.credentials = Credentials(**_load_env(env_file))

    return cfgs
