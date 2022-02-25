from abc import ABC, abstractmethod

import numpy as np


class StaticField(ABC):
    """
    Parent class for static electromagnetic fields.
    """

    def get_value(self, R: np.ndarray) -> np.ndarray:
        """
        Returns the value of the field at given position R.
        """

