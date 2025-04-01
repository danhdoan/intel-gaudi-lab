"""Averager Class.

This class is used to average the value of a set of points.
"""

__author__ = ["Danh Doan", "Nguyen Tran", "Hung Vo"]
__email__ = [
    "danh.doan@enouvo.com",
    "nguyen.tran@team.enouvo.com",
    "hung.vo@team.enouvo.com",
]
__date__ = "2023/02/28"
__status__ = "development"


# ======================================================================================


class Averager:
    """Averager Class."""

    def __init__(self, beta=0.99):
        """Init Averager object."""
        self.v = 0
        self.beta = beta
        self.t = 0

    def update(self, x):
        """Update v."""
        self.v = self.beta * self.v + (1 - self.beta) * x
        self.t += 1

    @property
    def value(self):
        """Get value of v."""
        return self.v / (1 - self.beta**self.t)


# ======================================================================================
