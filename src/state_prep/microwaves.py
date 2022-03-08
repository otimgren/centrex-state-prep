from dataclasses import dataclass
from typing import Callable, Union

import centrex_TlF
import numpy as np
from centrex_TlF.constants import constants_X
from centrex_TlF.couplings import calculate_ED_ME_mixed_state
from scipy import constants


@dataclass
class MWSpatialDependence:
    """
    Class for representing spatial dependence of microwave field intensity.
    """

    I_R: Callable
    polarization: Polarization

    def Omega_R(
        self, ground_main: centrex_TlF.State, excited_main: centrex_TlF.State,
    ):
        """
        Converts the intensity into Rabi rate.
        """
        pol_vec = self.polarization.

        angular_ME = calculate_ED_ME_mixed_state(
            excited_main, ground_main, pol_vec=pol_vec
        )

        return angular_ME * constants_X.D_TlF * E_R(R)

    def E_R(self, R: np.ndarray) -> np.ndarray:
        """
        Convert intensity (W/m^2) to electric field in V/cm and return electric
        field magnitude.
        """
        return np.sqrt(2 * I_R(R) / (constants.c * constants.epsilon_0)) / 100


@dataclass
class Polarization:
    """
    Class for representing polarization of microwave fields.
    """

    p_R_main: Callable
    k_vec: np.ndarray

    def get_long_pol(self):
        """
        Returns the longitudinal polarization component
        """

    def p_R(self, R:np.ndarray)-> np.ndarray:
        """
        Calculate polarization vector at given point and return it
        """


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
