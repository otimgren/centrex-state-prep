from dataclasses import dataclass
from typing import Callable, Union

import centrex_TlF
import numpy as np


@dataclass
class SpatialDependenceIntensity:
    """
    Class for representing spatial dependence of microwave field intensity.
    """

    integral: float
    I_R: Callable

    def convert_to_Rabi(self, ground_main, excited_main):
        """
        Converts the intensity into Rabi rate.
        """


@dataclass
class SpatialDependenceRabi:
    """
    Class for representing spatial dependence of microwave field Rabi rate.
    """

    integral: float
    Omega_R: Callable
    ground_main: centrex_TlF.State
    excited_main: centrex_TlF.State


@dataclass
class Polarization:
    """
    Class for representing polarization of microwave fields.
    """

    p_R: Callable
    k_vec: np.ndarray

    def get_long_pol(self):
        """
        Returns the longitudinal polarization component
        """
        pass


class MicrowaveField:
    """
    Class to represent microwave fields.
    """

    def __init__(
        self,
        Jg: int,
        Je: int,
        spatial_dep: Union[SpatialDependenceRabi, SpatialDependenceIntensity],
        pol: Polarization,
        ground_main: centrex_TlF.State = None,
        excited_main: centrex_TlF.State = None,
    ) -> None:
        self.Jg = Jg
        self.Je = Je
        self.spatial_dep = spatial_dep
        self.pol = pol
