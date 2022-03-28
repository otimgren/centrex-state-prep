from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class Trajectory:
    Rini: np.ndarray
    Vini: np.ndarray
    zfin: float

    def __post_init__(self):
        self.Rini = self.Rini  # .reshape(3, 1)
        self.Vini = self.Vini  # .reshape(3, 1)

    def R_t(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """
        Returns position along trajectory in XYZ coordinates at given time
        """
        # If t is an array, make sure it's a suitable shape
        if isinstance(t, np.ndarray):
            t = t.reshape(1, len(t))

        return self.Rini + t * self.Vini

    def get_T(self) -> float:
        """
        Return the total integration time required for molecule to get from
        zini (i.e. Rini[2]) to zfin
        """
        return (self.zfin - self.Rini[2]) / self.Vini[2]
