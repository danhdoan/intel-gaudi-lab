"""Main Application."""

__author__ = "Danh Doan"
__email__ = "danhdoancv@gmail.com"
__date__ = "2023/03/24"
__status__ = "development"


# ======================================================================================


from libs.cli import cli_parser
from libs.utils import logger_utils
from libs.utils.common import dbg

logger = logger_utils.setup_logger(name="app", log_dir="logs")


# ======================================================================================


def app():
    """Perform logic."""
    dbg("hello")


# ======================================================================================


def main():
    """Perform main logic."""
    _args = cli_parser.get_args()

    app()


# ======================================================================================


if __name__ == "__main__":
    logger.info("Task: Setup base project\n")

    main()

    logger.info("Process Done")


# ======================================================================================
