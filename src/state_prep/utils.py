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
