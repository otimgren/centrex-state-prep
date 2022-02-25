from centrex_TlF import State


def vector_to_state(state_vector, QN, E=None):
    state = State()

    # Get data in correct format for initializing state object
    for j, amp in enumerate(state_vector):
        state += amp * QN[j]

    return state
