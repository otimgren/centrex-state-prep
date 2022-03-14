from dataclasses import dataclass
from typing import Callable, List, Union

import centrex_TlF
import numpy as np
from centrex_TlF import State
from centrex_TlF.constants import constants_X
from centrex_TlF.couplings.matrix_elements import calculate_ED_ME_mixed_state
from scipy import constants
from scipy.misc import derivative


@dataclass
class Polarization:
    """
    Class for representing polarization of microwave fields.
    """

    p_R_main: Callable
    k_vec: np.ndarray

    def __post_init__(self):
        # Check that k-vector and polarization are orthogonal
        err_msg = "k-vector and main polarization should be orthogonal"
        assert (self.p_R_main(np.array((0, 0, 0))) @ self.k_vec) < 1e-6, err_msg

        # Check that k-vector is normalized
        err_msg = "k-vector not normalized"
        assert np.sum(np.abs(self.k_vec) ** 2) == 1.0, err_msg

    def get_long_pol(self, R: np.ndarray, E_R: Callable, freq: float) -> np.ndarray:
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
        k = 2 * np.pi * freq / constants.c

        # Scale polarization vector appropriately
        if k != 0:
            p_long = div / (-1j * k) * self.k_vec

        else:
            p_long = 0 * self.k_vec

        return p_long

    def p_R(self, R: np.ndarray, E_R: Callable = None, freq: float = 0) -> np.ndarray:
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
            p_long = self.get_long_pol(R, E_R, freq)

            # Normalize polarization vector
            p = p_main + p_long
            p = p / np.abs(p) ** 2

            return p


class Intensity:
    """
    Class for representing spatial dependence of microwave field intensity.
    """

    def __init__(self, I_R: Callable):
        self.I_R = I_R

    def E_R(self, R: np.ndarray, power: float = None) -> np.ndarray:
        """
        Convert intensity (W/m^2) to electric field in V/cm and return electric
        field magnitude.
        """
        return np.sqrt(2 * I_R(R, power) / (constants.c * constants.epsilon_0)) / 100


class MicrowaveField:
    """
    Class to represent microwave fields.
    """

    def __init__(
        self,
        Jg: int,
        Je: int,
        intensity: Intensity,
        polarization: Polarization,
        muW_freq: float,
    ) -> None:
        self.Jg = Jg  # J for ground state
        self.Je = Je  # J for excited state
        self.intensity = intensity  # Spatial dependence of microwave intensity
        self.polarization = polarization  # Polarization of microwave field
        self.muW_freq = muW_freq  # Frequency of microwaves
        self.H_list = None  # Couplings for x,y,z-polarized microwaves
        self.D = None  # Matrix for shifting energies in rotating frame

    def generate_coupling_matrices(self, QN: List[centrex_TlF.State]) -> None:
        """
        Generates list of coupling Hamiltonians for x,y,z polarized microwaves
        coupling Jg to Je in basis defined by QN. 
        """
        Jg = self.Jg
        Je = self.Je

        # Loop over possible polarizations and generate coupling matrices
        H_list = []
        for i in range(0, 3):
            # Generate polarization vector
            pol_vec = np.array([0, 0, 0])
            pol_vec[i] = 1

            # Generate coupling matrix
            H = make_H_mu(Je, Jg, QN, pol_vec=pol_vec)

            # Remove small components
            H[np.abs(H) < 1e-3 * np.max(np.abs(H))] = 0

            # Check that matrix is Hermitian
            is_hermitian = np.allclose(H, H.conj().T)
            if not is_hermitian:
                print(
                    "Warning: Microwave coupling matrix {} is not Hermitian!".format(i)
                )

            # Append to list of coupling matrices
            H_list.append(H)

        self.H_list = H_list

    def generate_D(self, QN: List[centrex_TlF.State]) -> None:
        """ 
        Generates a diagonal matrix that is used to shift energies in the rotating frame
        """
        Je = self.Je
        omega = 2 * np.pi * self.muW_freq

        # Generate the shift matrix
        D = np.zeros((len(QN), len(QN)))
        for i in range(len(QN)):
            if QN[i].J == Je:
                D[i, i] = -omega


def make_H_mu(J1, J2, QN, pol_vec=np.array((0, 0, 1))):
    """
    Function that generates Hamiltonian for microwave transitions between J1 and J2 (all hyperfine states) for given
    polarization of microwaves. Rotating wave approximation is applied implicitly by only taking the exp(+i*omega*t) part
    of the cos(omgega*t) into account
    
    inputs:
    J1 = J of the lower rotational state being coupled
    J2 = J of upper rotational state being coupled
    QN = Quantum numbers of each index (defines basis fo matrices and vectors)
    pol_vec = vector describing polarization 
    
    returns:
    H_mu = Hamiltonian describing coupling between 
    
    """
    # Figure out how many states there are in the system
    N_states = len(QN)

    # Initialize a Hamiltonian
    H_mu = np.zeros((N_states, N_states), dtype=complex)

    # Start looping over states and calculate microwave matrix elements between them
    for i in range(0, N_states):
        state1 = QN[i]

        for j in range(i, N_states):
            state2 = QN[j]

            # Check that the states have the correct values of J
            if (state1.J == J1 and state2.J == J2) or (
                state1.J == J2 and state2.J == J1
            ):
                # Calculate matrix element between the two states
                H_mu[i, j] = calculate_microwave_ME(
                    state1, state2, reduced=False, pol_vec=pol_vec
                )

    # Make H_mu hermitian
    H_mu = (H_mu + np.conj(H_mu.T)) - np.diag(np.diag(H_mu))

    # return the coupling matrix
    return H_mu


def calculate_microwave_power(
    state1: State,
    state2: State,
    Omega: float,
    R: np.ndarray,
    microwave_field: MicrowaveField,
) -> float:
    """
    Calculates the microwave power required to have Rabi rate Omega for the microwave
    transition between state1 and state2 at position R
    """

    # Calculate electric field magnitude at R for 1W of total power
    E = microwave_field.intensity.E_R(R, power=1.0)

    # Determine main polarization component of microwave field at given point
    pol_vec = microwave_field.polarization.p_R_main(R)

    # Calculate the angular part of the matrix element between the states
    ME = calculate_ED_ME_mixed_state(state1, state2, pol_vec=pol_vec)

    # Calculate the Rabi rate for P = 1W
    Omega1W = constants_X.D_TlF * E / 2

    # Determine what power is required (Omega \propto sqrt(Power))
    power_req = (Omega / Omega1W) ** 2

    return power_req

