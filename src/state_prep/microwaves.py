from dataclasses import dataclass
from typing import Callable, Union

import centrex_TlF
import numpy as np
from centrex_TlF.constants import constants_X
from centrex_TlF.couplings import calculate_ED_ME_mixed_state
from scipy import constants
from scipy.misc import derivative


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
        pol_vec = self.polarization.p_R(R, self.E_R)

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
    freq: float = None

    def __post_init__(self):
        # Check that k-vector and polarization are orthogonal
        err_msg = "k-vector and main polarization should be orthogonal"
        assert (self.p_R_main(np.array((0, 0, 0))) @ self.k_vec) < 1e-6, err_msg

        # Check that k-vector is normalized
        err_msg = "k-vector not normalized"
        assert np.abs(self.k_vec) ** 2 == 1.0, err_msg

    def get_long_pol(self, R: np.ndarray, E_R: Callable) -> np.ndarray:
        """
        Returns the longitudinal polarization component.
        """
        # Take div of E_R along main polarization
        div = 0
        for i in range(3):
            unit_vec = np.zeros((3, 1))
            unit_vec[i] = 1
            func = lambda x: E_R((R - unit_vec.T @ R) + x * unit_vec)
            div += derivative(func, R[i], dx=1e-3)

        # Calculate wavenumber for field
        k = 2 * np.pi * self.freq / constants.c

        # Scale polarization vector appropriately
        p_long = div / (-1j * k) * self.k_vec
        return p_long

    def p_R(self, R: np.ndarray, E_R: Callable = None) -> np.ndarray:
        """
        Calculate polarization vector at given point and return it
        """
        # If no spatial electric field, specified, ignore any longitudinal component
        # from spatial variation
        if not E_R:
            return self.p_R_main(R)

        else:
            # Calculate main component of polarization
            p_main = self.p_R_main(R)

            # Calculate longitudinal component
            p_long = self.get_long_pol(R, E_R)

            # Normalize polarization vector
            p = p_main + p_long
            p = p / np.abs(p) ** 2

            return p


class MicrowaveField:
    """
    Class to represent microwave fields.
    """

    def __init__(
        self,
        Jg: int,
        Je: int,
        spatial_dep: MWSpatialDependence,
        ground_main: centrex_TlF.State = None,
        excited_main: centrex_TlF.State = None,
    ) -> None:
        self.Jg = Jg
        self.Je = Je
        self.spatial_dep = spatial_dep
        self.pol = pol
