import numpy as np
from state import State

def gaussian(X, Y, amplitude=1.0, x0=0.5, y0=0.5, sigma=0.1, background=0.0):
    
    return background + amplitude * np.exp(
        -((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)
    )


def initialise(grid, config):

    b = np.zeros((config.ny, config.nx))

    D = gaussian(
        grid.X,
        grid.Y,
        amplitude=0.1,
        x0=config.Lx/2,
        y0=config.Ly/2,
        sigma=0.1,
        background=1.0
    )

    u = np.zeros((config.ny, config.nx))
    v = np.zeros((config.ny, config.nx))

    gamma = np.ones((config.ny, config.nx))

    # Conserved buoyancy density.
    Q = D * gamma

    state = State(
        D=D,
        u=u,
        v=v,
        Q=Q
    )

    return state, b