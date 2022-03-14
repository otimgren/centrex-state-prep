from dataclasses import dataclass
from typing import Callable, List

import centrex_TlF

from .trajectory import Trajectory


class Hamiltonian:
    pass


@dataclass
class SlowHamiltonian(Hamiltonian):
    """
    Representation for the slowly evolving Hamiltonian which contains the
    internal Hamiltonian of the TlF molecule and the Stark and Zeeman Hamiltonians.
    """

    Js: List[int]
    trajectory: Trajectory
    E_R: Callable
    B_R: Callable
    basis: str = "uncoupled"

    def __post_init__(self):
        if self.basis == "uncoupled":
            self.QN = centrex_TlF.states.generate_uncoupled_states_ground(self.Js)
            H_dict = centrex_TlF.hamiltonian.generate_uncoupled_hamiltonian_X(self.QN)
            self.H_EB = centrex_TlF.hamiltonian.generate_uncoupled_hamiltonian_X_function(
                H_dict
            )
        else:
            NotImplementedError("Basis not implemented.")

        self.H_R = lambda R: self.H_EB(self.E_R(R), self.B_R(R))

    def get_H_t_func(self) -> Callable:
        """
        Returns a function that gives the slow Hamiltonian as a function of time.
        """
        return lambda t: self.H_R(self.trajectory.R_t(t))

