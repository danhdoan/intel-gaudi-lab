"""Timer Utility."""

__author__ = ["Danh Doan", "Nguyen Tran", "Hung Vo"]
__email__ = [
    "danh.doan@enouvo.com",
    "nguyen.tran@team.enouvo.com",
    "hung.vo@team.enouvo.com",
]
__date__ = "2023/02/28"
__status__ = "development"


# ======================================================================================


import logging
import time

logger = logging.getLogger()


# ======================================================================================


def tiktok(func):
    """Decorate input function to measure running time.

    Args:
    ----
    func (object) : function to be decorated


    Returns:
    -------
    (object) : function after decorated
    """

    def inner(*args, **kwargs):
        """Calculate time."""
        begin = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        time_taken_ms = (end - begin) * 1000
        logger.debug(f"Time taken for {func.__name__}: {time_taken_ms:.5f} ms")
        return result

    return inner


# ======================================================================================
