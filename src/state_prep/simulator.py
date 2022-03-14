from dataclasses import dataclass
from typing import Callable, List, Tuple

import centrex_TlF
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg.lapack import zheevd
from tqdm import tqdm

from .electric_fields import ElectricField
from .hamiltonians import Hamiltonian, SlowHamiltonian
from .magnetic_fields import MagneticField
from .microwaves import MicrowaveField
from .trajectory import Trajectory
from .utils import find_max_overlap_idx, reorder_evecs, vector_to_state


@dataclass
class SimulationResult:
    """
    Class for storing results from simulations.

    t_array         : times at which results were returned
    psis            : state vectors at each time when starting from initial states defined in 
                      initial_states
    energies        : energies of all eigenstates of the hamiltonian at each time
    probabilities   : for each state in initial staets, the probability of being found in 
                      each eigenstate of the hamiltonian
    V_ref           : reference matrix of eigenstates of hamiltonian which tells what state each
                      index of energies and states corresponds to
    """

    trajectory: Trajectory
    electric_field: ElectricField
    magnetic_field: MagneticField
    initial_states: List[centrex_TlF.State]
    hamiltonian: Hamiltonian
    microwave_fields: List[MicrowaveField]
    t_array: np.ndarray
    psis: np.ndarray
    energies: np.ndarray
    energies_diabatic: np.ndarray
    probabilities: np.ndarray
    V_ini: np.ndarray
    V_fin: np.ndarray

    def plot_state_probability(
        self,
        state: centrex_TlF.State,
        initial_state: centrex_TlF.State,
        ax: plt.Axes = None,
        tolerance: float = 0.5,
    ) -> None:
        """
        Plots the probability of being found in a given adiabatically evolved eigenstate
        at different times.
        """
        if ax is None:
            fig, ax = plt.subplots()

        probs = self.get_state_probability(state, initial_state)
        label = (
            state.remove_small_components(tol = tolerance).normalize().make_real().__repr__()
        )
        ax.plot(self.t_array / 1e-6, probs, label=label)
        ax.set_xlabel(r"Time / $\mu$s")

    def plot_state_probabilities(
        self,
        states: List[centrex_TlF.State],
        initial_state: centrex_TlF.State,
        ax: plt.Axes = None,
        tolerance: float = 0.5
    ) -> None:
        """
        Plots probabilities over time for states specified in the list states.
        """
        if ax is None:
            fig, ax = plt.subplots()
        for state in states:
            self.plot_state_probability(state, initial_state, ax=ax, tolerance = tolerance)

    def get_state_probability(
        self, state: centrex_TlF.State, initial_state: centrex_TlF.State, ax=None
    ):
        """
        Returns the probability of being found in given adiabatically evolved state
        for given initial state.
        """
        index_ini = self.initial_states.index(initial_state)

        index_state = find_max_overlap_idx(
            state.state_vector(self.hamiltonian.QN), self.V_ini
        )

        return self.probabilities[:, index_ini, index_state]

    def find_large_prob_states(
        self, initial_state: centrex_TlF.State, N: int = 5
    ) -> List[centrex_TlF.State]:
        """
        Returns the N states with the largest mean probabilities for given initial
        state.
        """
        index_ini = self.initial_states.index(initial_state)
        index = np.argsort(-np.mean(self.probabilities[:, index_ini, :], axis=0))[:N]

        state_vecs = self.V_ini[:, index]
        states = []
        for i in range(state_vecs.shape[1]):
            states.append(vector_to_state(state_vecs[:, i], self.hamiltonian.QN))

        return states

    def plot_state_energy(
        self,
        state: centrex_TlF.State,
        zero_state: centrex_TlF = None,
        ax: plt.Axes = None,
    ):
        """
        Plots the energy of state, using energy of zero_state as zero energy.
        """
        energies = self.get_state_energy(state)
        if zero_state:
            energies_zero = self.get_state_energy(zero_state)
            energies = energies - energies_zero

        if ax is None:
            fig, ax = plt.subplots()

        label = (
            state.remove_small_components(tol=0.5).normalize().make_real().__repr__()
        )
        ax.plot(self.t_array / 1e-6, energies / (2 * np.pi / 1e3), label=label)
        ax.set_xlabel(r"Time / $\mu$s")
        ax.set_ylabel("Energy / kHz")

    def plot_state_energies(
        self,
        states: List[centrex_TlF.State],
        zero_state: centrex_TlF.State,
        ax: plt.Axes = None,
    ) -> None:
        """
        Plots probabilities over time for states specified in the list states.
        """
        if ax is None:
            fig, ax = plt.subplots()
        for state in states:
            self.plot_state_energy(state, zero_state=zero_state, ax=ax)

    def get_state_energy(self, state: centrex_TlF.State) -> np.ndarray:
        """
        Gets the energy of state for all values in t_array.
        """

        index_state = find_max_overlap_idx(
            state.state_vector(self.hamiltonian.QN), self.V_ini
        )

        return self.energies[:, index_state]

    def get_state_energy_diabatic(self, state: centrex_TlF.State) -> np.array:
        """
        Gets the energy of state that is closes to provided state at each time step.

        Corresponds (somewhat) to diabatic following of eigenstates
        """

        index_state = find_max_overlap_idx(
            state.state_vector(self.hamiltonian.QN), self.V_ini
        )

        return self.energies_diabatic[:, index_state]


@dataclass
class Simulator:
    """
    Class used to run simulations and store data.
    """

    trajectory: Trajectory
    electric_field: ElectricField
    magnetic_field: MagneticField
    initial_states_approx: centrex_TlF.State
    hamiltonian: Hamiltonian
    microwave_fields: List[MicrowaveField] = None

    def __post_init__(self):
        self.psis = None
        self.initial_states = None

    def run(self, N_steps=int(1e4)):
        """
        Runs the simulation.
        """
        # Calculate the total time for the simulation
        T = self.trajectory.get_T()

        # Make a function of electric field over time
        E_t = self.electric_field.get_E_t_func(self.trajectory.R_t)

        # Function of B-field over time
        B_t = self.magnetic_field.get_B_t_func(self.trajectory.R_t)

        # Generate Hamiltonian that has slow time-evolution included
        H_t = self.hamiltonian.get_H_t_func()

        # Generate Hamiltonians for microwaves
        if self.microwave_fields is not None:
            # Initialize matrix for shifting energies in rotating frame
            D_mu = np.zeros(len(self.hamiltonian.QN), len(self.hamiltonian.QN))

            # Initialize container for Hamiltonians
            muw_hams = []

            for microwave_field in self.microwave_fields:
                muw_hams.append(
                    microwave_field.get_H_t_func(
                        self.trajectory.R_t, self.hamiltonian.QN
                    )
                )
                D_mu += microwave_field.D

        # Generate function that gives couplings due to all microwaves
        def H_mu_tot_t(t):
            H_mu_tot = muw_hams[0](t)
            if len(muw_hams) > 1:
                for H_mu_t in muw_hams[1:]:
                    H_mu_tot = H_mu_tot + H_mu_t(t)
            return H_mu_tot

        # Generate time array
        t_array = np.linspace(0, T, N_steps)

        # Perform time-evolution
        if self.microwave_fields is None:
            (
                psis_t,
                energies,
                energies_diabatic,
                probalities,
                V_ini,
                V_fin,
            ) = self._time_evolve(H_t, t_array)
        else:
            psis_t, energies, probalities, V_ini, V_fin = self._time_evolve_mu(
                H_t, H_mu_t, t_array
            )

        # Generate a result object
        result = SimulationResult(
            self.trajectory,
            self.electric_field,
            self.magnetic_field,
            self.initial_states,
            self.hamiltonian,
            self.microwave_fields,
            t_array,
            psis_t,
            energies,
            energies_diabatic,
            probalities,
            V_ini,
            V_fin,
        )

        return result

    def init_state_vecs(self, H_0) -> None:
        """
        Generates state vectors based on self.initial_states in the basis 
        of self.hamiltonian
        """

        # Find the eigenstates of the Hamiltonian that most closely correspond to the
        # initial states
        self.initial_states = []
        self.psis = []
        _, V = np.linalg.eigh(H_0)
        for state in self.initial_states_approx:
            idx = centrex_TlF.states.find_state_idx_from_state(
                H_0, state, self.hamiltonian.QN
            )
            self.psis.append(V[:, idx])
            self.initial_states.append(vector_to_state(V[:, idx], self.hamiltonian.QN))

    def _time_evolve(self, H_slow: Callable, t_array: np.ndarray):
        """
        Time evolves the system using the Hamiltonian function H_t
        over the time period in t_array.
        """
        # Calculate Hamiltonian at tini
        H_tini = H_slow(t_array[0])

        # Initialize state vectors
        self.init_state_vecs(H_tini)

        # Initialize containers to store results
        psis_t, energies, probabilities = self._init_results_containers(t_array, H_tini)
        energies_diabatic = energies.copy()

        # Initialize reference matrix of eigenvectors that is used to keep track
        # of adiabatic evolution of eigenstates
        E_ref, V_ref = np.linalg.eigh(H_tini)
        V_ref_ini = V_ref

        # Loop over t_array to time-evolve
        for i, t in enumerate(tqdm(t_array[:-1])):
            # Calculate the timestep
            dt = t_array[i + 1] - t_array[i]

            # Calculate Hamiltonian
            H_slow_i = H_slow(t)

            # Diagonalize Hamiltonian
            D, V, info = zheevd(H_slow_i)
            if info != 0:
                D, V = np.linalg.eigh(H_slow_i)

            # Reorder eigenvectors and energies
            Es, evecs = reorder_evecs(V, D, V_ref)
            Es_diabatic, _ = reorder_evecs(V, D, V_ref_ini)

            # Calculate propagator for the system
            U_dt = V @ np.diag(np.exp(-1j * D * dt)) @ V.conj().T

            # Apply propagator to each state vector
            self.psis = np.einsum("ij,kj->ki", U_dt, self.psis)

            # Store results for this timestep
            psis_t[i + 1, :, :] = self.psis
            energies[i + 1, :] = Es
            energies_diabatic[i + 1, :] = Es_diabatic
            probabilities[i + 1, :, :] = self.calculate_probabilities(self.psis, evecs)

            # Change V_ref
            V_ref = evecs

        return psis_t, energies, energies_diabatic, probabilities, V_ref_ini, V_ref

    def _time_evolve_mu(self, H_slow: Callable, H_mu: Callable, t_array: np.ndarray):
        """
        Time evolves the system using the Hamiltonian function H_t
        over the time period in t_array.
        """
        # Calculate Hamiltonian at tini
        H_tini = H_slow(t_array[0])

        # Initialize state vectors
        self.init_state_vecs(H_tini)

        # Initialize containers to store results
        psis_t, energies, probabilities = self._init_results_containers(t_array, H_tini)

        # Initialize reference matrix of eigenvectors that is used to keep track
        # of adiabatic evolution of eigenstates
        E_ref, V_ref = np.linalg.eigh(H_tini)
        V_ref_ini = V_ref

        # Loop over t_array to time-evolve
        for i, t in enumerate(tqdm(t_array[:-1])):
            # Calculate the timestep
            dt = t_array[i + 1] - t_array[i]

            # Calculate Hamiltonians
            H_slow_i = H_slow(t)
            H_mu_i = H

            # Diagonalize Hamiltonian
            D, V, info = zheevd(H_slow_i)
            if info != 0:
                D, V = np.linalg.eigh(H_slow_i)

            # Reorder eigenvectors and energies
            Es, evecs = reorder_evecs(V, D, V_ref)

            # Calculate propagator for the system
            U_dt = V @ np.diag(np.exp(-1j * D * dt)) @ V.conj().T

            # Apply propagator to each state vector
            self.psis = np.einsum("ij,kj->ki", U_dt, self.psis)

            # Store results for this timestep
            psis_t[i + 1, :, :] = self.psis
            energies[i + 1, :] = D
            probabilities[i + 1, :, :] = self.calculate_probabilities(self.psis, evecs)

            # Change V_ref
            V_ref = evecs

        return psis_t, energies, probabilities, V_ref_ini, V_ref

    def _init_results_containers(self, t_array: np.ndarray, H_tini: np.ndarray):
        """
        Initializes containers for time evolution results based on array of times
        and Hamiltonian at initial time.
        """
        # Storage for state vectors
        psis_t = np.zeros(
            (len(t_array), len(self.initial_states), len(self.hamiltonian.QN)),
            dtype="complex",
        )
        psis_t[0, :, :] = self.psis

        # Storage for energies
        energies = np.zeros((len(t_array), len(self.hamiltonian.QN)))

        # Storage for state probabilities
        probabilities = np.zeros(psis_t.shape)

        # Calculate values for energies and probabilities at t_ini
        D, V = np.linalg.eigh(H_tini)

        # Store values
        energies[0, :] = D
        probabilities[0, :, :] = self.calculate_probabilities(self.psis, V)

        return psis_t, energies, probabilities

    def calculate_probabilities(self, psis: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Given state vectors as columns of psi, for each state vector, returns the
        probabilities of being in states stored as columns of V.
        """
        overlaps = np.einsum("ij,kj->ki", V.conj().T, psis)
        return np.abs(overlaps) ** 2

