from typing import Tuple

import numpy as np
from centrex_TlF import State


def vector_to_state(state_vector, QN, E=None):
    state = State()

    # Get data in correct format for initializing state object
    for j, amp in enumerate(state_vector):
        state += amp * QN[j]

    return state


def find_max_overlap_idx(state_vec: np.ndarray, V_matrix: np.ndarray) -> int:
    # Take dot product between each eigenvector in V and state_vec
    overlap_vectors = np.absolute(V_matrix.conj().T @ state_vec)

    # Find index of state that has the largest overlap
    index = np.argmax(overlap_vectors)

    return index


def reorder_evecs(
    V_in: np.ndarray, E_in: np.ndarray, V_ref: np.ndarray
) -> Tuple[np.ndarray]:
    """
    Reorders the matrix of eigenvectors V_in (each column is an eigenvector) and
    corresponding energies E_in so that the eigenvector overlap between V_in and V_ref
    is maximised.
    """
    # Take dot product between each eigenvector in V and state_vec
    overlap_vectors = np.absolute(np.matmul(np.conj(V_in.T), V_ref))

    # Find which state has the largest overlap:
    index = np.argsort(np.argmax(overlap_vectors, axis=1))
    # Store energy and state
    E_out = E_in[index]
    V_out = V_in[:, index]

    return E_out, V_out
