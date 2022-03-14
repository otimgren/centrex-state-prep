"""
Functions and classes for plotting.
"""
from dataclasses import dataclass
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from centrex_TlF import State
from centrex_TlF.constants.constants_X import D_TlF

from .hamiltonians import Hamiltonian
from .microwaves import MicrowaveField
from .utils import find_max_overlap_idx, reorder_evecs


@dataclass
class CouplingPlotter:
    state_pairs: List[Tuple[State]]
    E_t: Callable
    H_t: Callable
    QN: List[State]
    t_array: np.ndarray

    def __post_init__(self) -> None:
        # Find eigenstates of H_t at each time
        self.find_eigenstates()

        # Make sure the microwave_field coupling Hamiltonian has been generated
        if self.microwave_field.H_list is None:
            raise ValueError("No microwave_field coupling matrices found")

    def plot_coupling_strengths(self, ax=None):
        """
        Plots the coupling strengths for all the state pairs. 
        """
        if ax is None:
            fig, ax = plt.subplots()

        for couplings, state_pair in zip(self.MEs, self.state_pairs):
            label = f"{state_pair[0].__repr__} <-> {state_pair[1].__repr__}"
            self.plot_coupling_strength(couplings, ax, label=label)

    def plot_coupling_strength(
        self, couplings: np.ndarray, ax: plt.Axes, label: str = None
    ):
        """
        Plots the coupling strength for a single state pair
        """
        ax.plot(self.t_array, couplings, label=label)

    def find_eigenstates(self) -> None:
        """
        Finds the eigenstates of H_t at each time in t_array while making sure they are
        ordered consistently (adiabatic following of states)
        """
        # Initialize container for eigenstates at each time
        V_array = np.zeros((len(self.t_array, len(self.QN), len(self.QN))))

        D_ref_0, V_ref_0 = np.linalg.eigh(H)
        self.V_ref = V_ref_0
        V_ref = V_ref_0

        for i, t in enumerate(t_array):
            D, V = np.linalg.eigh(H_t(t))
            D, V = reorder_evecs(V, E, V_ref)
            V_array[i, :, :] = V

        self.V_array = V_array

    def find_couplings(self) -> None:
        """
        Find the matrix elements due to microwave_field between state_pairs
        at different times
        """

        # Find the state vectors for each state in at each time
        self.state_pair_vecs = self._find_exact_state_vecs()

        # Calculate matrix elements between state pairs at each time
        self.MEs = self._calculate_matrix_elements()

    def _find_state_vecs(self, state) -> np.ndarray:
        """
        Finds exact state vecs for a state at all times in t_array 
        """
        # Find the index that corresponds to the state
        idx = find_max_overlap_idx(state.state_vec(self.QN), self.V_ref)

        return self.V_array[:, :, idx]

    def _find_exact_state_vec_pairs(self) -> None:
        """
        Finds exact eigenstates of H_t that correspond to states in state_pairs
        at all times
        """
        vec_pairs = []
        for state_pair in self.state_pairs:
            vec_pairs.append(
                (
                    self._find_state_vecs(state_pair[0]),
                    self._find_state_vecs(state_pair[1]),
                )
            )

        return vec_pairs

    def _calculate_matrix_elements(self) -> List[np.ndarray]:
        """
        Returns matrix element between each pair of states at eact time in 
        t_array
        """
        # List for storing results
        couplings_list = []

        # Get coupling matrices for each polarization
        H_mu_x = self.microwave_field.H_list[0]
        H_mu_y = self.microwave_field.H_list[1]
        H_mu_z = self.microwave_field.H_list[2]

        for state_vecs1, state_vecs2 in self.state_pair_vecs:
            couplings = np.zeros(self.t_array.shape)

            for i, t in enumerate(t_array):
                E_t = self.E_t(t)
                couplings[i] = D_TlF * (
                    E_t[0] * H_mu_x + E_t[1] * H_mu_y + E_t[2] * H_mu_z
                )

            couplings_list.append(couplings)

        return couplings_list
