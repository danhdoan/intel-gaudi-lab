"""Common Utilities.

Common Support Functions
"""

__author__ = ["Danh Doan", "Nam Ho"]
__email__ = ["danh.doan@enouvo.com", "nam.ho@enouvo.com"]
__date__ = "2024/11/20"
__status__ = "development"


# ======================================================================================


import json
import yaml

from typing import Dict, List, Optional


# ======================================================================================


def load_json(json_path: str, encoding: str = "utf-8") -> List | Dict:
    """Load data from JSON file.

    Args:
    ----
    json_path (str): Path to input JSON file
    encoding (str): Type of encoding
        (default: "uft-8")

    Returns:
    -------
    data (List | Dist) : Data loaded from JSON file

    """
    data = None
    with open(json_path, encoding=encoding) as f:
        data = json.load(f)

    return data


# ======================================================================================


def save_json(
    json_path: str, data: Dict, indent: str = 2, ensure_ascii: bool = False
) -> None:
    """Serialize data to JSON file.

    Args:
    ----
    json_path (str): Path of output JSON file
    data (Dict): Content to serialize
    indent (int): Set default indentation
        (default: 2)
    ensure_ascii (bool): Set output content as it is
        (default: False)

    """
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


# ======================================================================================


def to_json_str(data: List | Dict, convert_to_str: bool = True) -> str:
    """Serialize JSON style data to string.

    Args:
    ----
    data (List | Dict): JSON style data
    convert_to_str (bool): Option to stringify data values to readable string
        (eg. datetime.date(yyyy, mm, dd) -> "yyyy-mm-dd")
        (default: True)

    Returns:
    -------
    json_str (str): JSON style string

    """
    if convert_to_str:
        json_str = json.dumps(data, indent=2, default=str)
    else:
        json_str = json.dumps(data, indent=2)

    return json_str


# ======================================================================================


def print_json(data: List | Dict, convert_to_str: bool = True) -> None:
    """Prettify and print JSON style data.

    Args:
    ----
    data (List | Dict): JSON style data
    convert_to_str (bool): Option to stringify data values to readable string
        (eg. datetime.date(yyyy, mm, dd) -> "yyyy-mm-dd")
        (default: True)

    """
    print(to_json_str(data, convert_to_str))


# ======================================================================================


def load_yaml(yaml_file: str, encoding="utf-8") -> Dict[str, Optional[str]]:
    """Load data from YAML file.

    Args:
    ----
    yaml_file (str) : Path to YAML file
    encoding (str) : Type of encoding
        (default: "utf-8")

    Returns:
    -------
    data (Dict[str, Optional[str]]): Data in key-value format

    """
    try:
        with open(yaml_file, encoding=encoding) as file:
            data = yaml.safe_load(file)
            return data
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML data: {e}") from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{yaml_file} does not exist") from e
    except Exception as e:
        raise Exception(f"Error occured: {e}") from e


# ======================================================================================
