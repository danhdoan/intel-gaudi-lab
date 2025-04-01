"""Common Utilities.

Common Support functions
"""

__author__ = "Danh Doan"
__email__ = "danhdoancv@gmail.com"
__date__ = "2020/04/19"
__status__ = "development"


# ======================================================================================


import inspect
import json
import logging
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ======================================================================================


DEBUG_LENGTH = 75
DEBUG_SEPERATOR = "="


# ======================================================================================


def dbg(*args):
    """Debug by print values.

    Args:
    ----
    args (obj) : packed list of values to debug
    """
    caller = inspect.stack()[1]
    caller_module = inspect.getmodule(caller[0]).__name__
    caller_function, caller_line_number = caller.function, caller.lineno
    dbg_prefix = f"[{caller_module}-{caller_function}:{caller_line_number}] "
    logging.debug(dbg_prefix.ljust(DEBUG_LENGTH, DEBUG_SEPERATOR))

    for arg in args:
        if isinstance(arg, list):
            print_list(arg)
        else:
            logging.debug(arg)

    logging.debug("")


def print_list(lst):
    """Print items of a list.

    Args:
    ----
    lst (List[obj]) : list of object to print
    """
    for i, item in enumerate(lst):
        logging.debug("%d %s", i + 1, item)


# ======================================================================================


def load_json(json_path, encoding="utf-8"):
    """Load data from JSON file.

    Args:
    ----
    json_path (str) : path to input JSON file
    encoding (Str): type of encoding

    Returns:
    -------
    (obj) : data loaded from JSON file
    """
    data = None
    with open(json_path, encoding=encoding) as f:
        data = json.load(f)

    return data


# ======================================================================================


def save_json(json_path, data, indent=2, ensure_ascii=False):
    """Serialize data to JSON file.

    Args:
    ----
    json_path (Str): path of output JSON file
    data (Dict): content to serialize
    indent (Int): set default indentation
    ensure_ascii (Bool): set output content is as-is
    """
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


# ======================================================================================


def load_yaml(yaml_file: str, encoding="utf-8") -> Dict[str, Optional[str]]:
    """Load data from YAML file.

    Args:
    ----
    yaml_file (str) : path to YAML file
    encoding (str) : type of encoding method

    Returns:
    -------
    data (Dict): data in key-value format
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
