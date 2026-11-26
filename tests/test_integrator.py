import numpy as np

from state import State
from time_evolution import ssprk3_step


def test_ssprk3_third_order_accuracy():

    def rhs(state):
        return State(
            D=-state.D,
            u=-state.u,
            v=-state.v,
            Q=-state.Q
        )

    def calculate_error(dt):
        state = State(
            D=np.array([[1.0]]),
            u=np.array([[1.0]]),
            v=np.array([[1.0]]),
            Q=np.array([[1.0]])
        )

        t_end = 1.0
        nsteps = round(t_end / dt)

        for _ in range(nsteps):
            state = ssprk3_step(state, dt, rhs)

        exact = np.exp(-t_end)

        return abs(state.D[0, 0] - exact)

    error_coarse = calculate_error(0.1)
    error_fine = calculate_error(0.05)

    error_ratio = error_coarse / error_fine

    assert 6.0 < error_ratio < 10.0