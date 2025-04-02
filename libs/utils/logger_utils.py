"""Logger Utility."""

__author__ = ["Danh Doan", "Nguyen Tran", "Hung Vo"]
__email__ = [
    "danh.doan@enouvo.com",
    "nguyen.tran@team.enouvo.com",
    "hung.vo@team.enouvo.com",
]
__date__ = "2023/02/28"
__status__ = "development"


# ==============================================================================


import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

# ==============================================================================


FORMATTER = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")


# ==============================================================================


def remove_handler_from_logger(logger, handler_type):
    """Remove handler from logger if exist."""
    for handler in logger.handlers:
        if isinstance(handler, handler_type):
            logger.removeHandler(handler)
            break


# ==============================================================================


def setup_stream_handler(
    logger,
    use_stream,
    logger_types,
    formatter=FORMATTER,
):
    """Set up stream handler for logger with specific condition."""
    if not use_stream:
        if logging.StreamHandler in logger_types:
            remove_handler_from_logger(logger, logging.StreamHandler)
        return

    if logging.StreamHandler not in logger_types:
        std_handler = logging.StreamHandler(sys.stdout)
        std_handler.setFormatter(formatter)
        logger.addHandler(std_handler)


# ==============================================================================


def setup_file_handler(  # noqa: PLR0913
    logger, name, log_dir, use_file, logger_types, formatter=FORMATTER
):
    """Set up file handler for logger with specific condition."""
    if not use_file:
        if TimedRotatingFileHandler in logger_types:
            remove_handler_from_logger(logger, TimedRotatingFileHandler)
        return

    if TimedRotatingFileHandler not in logger_types:
        file_handler = TimedRotatingFileHandler(
            os.path.join(log_dir, f"{name}.log"), when="midnight"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


# ==============================================================================


def setup_logger(
    name, log_dir="logs", use_stream=True, use_file=True, level=logging.DEBUG
):
    """Create logger with requested conditions."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()

    logger_types = set(type(handler) for handler in logger.handlers)

    setup_stream_handler(logger, use_stream, logger_types)
    setup_file_handler(logger, name, log_dir, use_file, logger_types)

    logger.setLevel(level)

    return logger


# ==============================================================================
