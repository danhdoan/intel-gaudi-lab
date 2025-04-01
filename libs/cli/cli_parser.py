"""CLI Parser.

Parse CLI arguments
"""

__author__ = "Danh Doan"
__email__ = "danhdoancv@gmail.com"
__date__ = "2021/08/20"
__status__ = "development"


# ======================================================================================


import argparse

# ======================================================================================


def get_args():
    """Parse CLI arguments from user.

    Returns
    -------
    (object) : parsed CLI arguments
    """
    parser = argparse.ArgumentParser()

    return parser.parse_args()


# ======================================================================================
