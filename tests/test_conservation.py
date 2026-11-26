import numpy as np

from config import ModelConfig
from grid import Grid
from model import Model
from time_evolution import ssprk3_step
from initial_conditions import initialise


def calculate_mass(state, config):
    return np.sum(state.D) * config.dx * config.dy


def test_mass_conservation_long_simulation():

    config = ModelConfig(
        nx=32,
        ny=32,
        dt=1e-4,
        t_end=0.1
    )

    grid = Grid(config)
    state, b = initialise(grid, config)
    model = Model(config, b)

    initial_mass = calculate_mass(state, config)

    nsteps = round(config.t_end / config.dt)

    for _ in range(nsteps):
        state = ssprk3_step(state, config.dt, model.rhs)

    final_mass = calculate_mass(state, config)

    assert np.isclose(
        final_mass,
        initial_mass,
        rtol=1e-10,
        atol=1e-12
    )

def calculate_total_buoyancy(state, config):
    return np.sum(state.Q) * config.dx * config.dy


def test_total_buoyancy_conservation():

    config = ModelConfig(
        nx=32,
        ny=32,
        dt=1e-4,
        t_end=0.1
    )

    grid = Grid(config)
    state, b = initialise(grid, config)
    model = Model(config, b)

    # Give buoyancy a spatial variation.
    gamma = (
        1.0
        + 0.1 * np.sin(2 * np.pi * grid.X / config.Lx)
    )

    state.Q = state.D * gamma
    initial_buoyancy = calculate_total_buoyancy(state, config)

    nsteps = round(config.t_end / config.dt)

    for _ in range(nsteps):
        state = ssprk3_step(state, config.dt, model.rhs)

    final_buoyancy = calculate_total_buoyancy(state, config)

    assert np.isclose(
        final_buoyancy,
        initial_buoyancy,
        rtol=1e-8,
        atol=1e-10
    )
