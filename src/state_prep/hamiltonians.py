from dataclasses import dataclass
from typing import Callable, List

import centrex_TlF


class Hamiltonian:
    pass


@dataclass
class SlowHamiltonian(Hamiltonian):
    Js: List[int]
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

    def get_H_t_func(self, E_t: Callable, B_t: Callable) -> Callable:
        """
        Returns a function that gives the Stark and Zeeman Hamiltonians 
        as function of time.
        """
        return lambda t: self.H_EB(E_t(t), B_t(t))

