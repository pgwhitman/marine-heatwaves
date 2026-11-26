import numpy as np
import pytest

from config import ModelConfig
from grid import Grid
from state import State
from model import Model
from time_evolution import ssprk3_step

from operators import (
    cell_to_xface,
    cell_to_yface,
    xface_to_cell,
    yface_to_cell,
)


def setup_model(epsilon=0.1):
    """Create a small model for testing."""

    config = ModelConfig(
        nx=32,
        ny=32,
        epsilon=epsilon,
        f0=1.0,
        dt=1.0e-4,
        t_end=0.01,
    )

    grid = Grid(config)

    b = np.zeros((config.ny, config.nx))

    model = Model(config, b)

    return config, grid, model


def make_state(config, grid, gaussian_bump=False):
    """Create a simple initial state."""

    if gaussian_bump:
        D = 1.0 + 0.05 * np.exp(
            -((grid.X - config.Lx/2)**2
              + (grid.Y - config.Ly/2)**2) / 0.05
        )
    else:
        D = np.ones((config.ny, config.nx))

    u = np.zeros((config.ny, config.nx))
    v = np.zeros((config.ny, config.nx))
    Q = np.ones((config.ny, config.nx))


    return State(D=D, u=u, v=v, Q=Q)


def calculate_mass(state, config):
    """Calculate total fluid mass."""

    return np.sum(state.D) * config.dx * config.dy


# --------------------------------------------------
# 1. Resting fluid
# --------------------------------------------------

def test_resting_fluid():
    """A uniform fluid at rest should remain at rest."""

    config, grid, model = setup_model()

    state = make_state(config, grid)

    derivative = model.rhs(state)

    assert np.allclose(derivative.D, 0.0)
    assert np.allclose(derivative.u, 0.0)
    assert np.allclose(derivative.v, 0.0)
    assert np.allclose(derivative.Q, 0.0)


# --------------------------------------------------
# 2. RHS output
# --------------------------------------------------

def test_rhs_shapes():
    """All RHS arrays should have the expected shape."""

    config, grid, model = setup_model()

    state = make_state(config, grid)
    derivative = model.rhs(state)

    expected_shape = (config.ny, config.nx)

    assert derivative.D.shape == expected_shape
    assert derivative.u.shape == expected_shape
    assert derivative.v.shape == expected_shape
    assert derivative.Q.shape == expected_shape


def test_rhs_finite():
    """The RHS should not contain NaNs or infinities."""

    config, grid, model = setup_model()

    state = make_state(config, grid, gaussian_bump=True)
    derivative = model.rhs(state)

    assert np.all(np.isfinite(derivative.D))
    assert np.all(np.isfinite(derivative.u))
    assert np.all(np.isfinite(derivative.v))
    assert np.all(np.isfinite(derivative.Q))


# --------------------------------------------------
# 3. Mass conservation
# --------------------------------------------------

def test_mass_conservation_rhs():
    """
    The spatial integral of D_t should be zero
    for periodic boundary conditions.
    """

    config, grid, model = setup_model()

    state = make_state(config, grid, gaussian_bump=True)
    derivative = model.rhs(state)

    mass_rate = (
        np.sum(derivative.D) * config.dx * config.dy
    )

    assert np.isclose(mass_rate, 0.0, atol=1.0e-12)


def test_mass_conservation_time_integration():
    """Total mass should remain constant during SSP-RK3."""

    config, grid, model = setup_model()

    state = make_state(config, grid, gaussian_bump=True)

    initial_mass = calculate_mass(state, config)

    nsteps = 20

    for _ in range(nsteps):
        state = ssprk3_step(state, config.dt, model.rhs)

    final_mass = calculate_mass(state, config)

    assert np.isclose(
        final_mass,
        initial_mass,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


# --------------------------------------------------
# 4. Constant buoyancy
# --------------------------------------------------

def test_constant_buoyancy_rhs():
    """Uniform gamma should satisfy Q_t = gamma_0 * D_t."""

    config, grid, model = setup_model()

    state = make_state(config, grid, gaussian_bump=True)

    gamma0 = 1.0
    derivative = model.rhs(state)

    assert np.allclose(
        derivative.Q,
        gamma0 * derivative.D
    )

def test_constant_buoyancy_after_evolution():
    """Uniform gamma should remain uniform during evolution."""

    config, grid, model = setup_model()

    state = make_state(config, grid, gaussian_bump=True)

    initial_gamma = state.Q / state.D

    for _ in range(20):
        state = ssprk3_step(state, config.dt, model.rhs)

    final_gamma = state.Q / state.D

    assert np.allclose(
        final_gamma,
        initial_gamma,
        atol=1e-10
    )


# --------------------------------------------------
# 5. Interpolation operators
# --------------------------------------------------

def test_interpolation_preserves_constant():
    """Interpolation should preserve a constant field."""

    q = np.full((32, 32), 3.5)

    assert np.allclose(cell_to_xface(q), 3.5)
    assert np.allclose(cell_to_yface(q), 3.5)
    assert np.allclose(xface_to_cell(q), 3.5)
    assert np.allclose(yface_to_cell(q), 3.5)


# --------------------------------------------------
# 6. Positive depth
# --------------------------------------------------

def test_positive_depth_after_evolution():
    """Depth should remain positive during a short simulation."""

    config, grid, model = setup_model()

    state = make_state(config, grid, gaussian_bump=True)

    for _ in range(20):
        state = ssprk3_step(state, config.dt, model.rhs)

    assert np.all(state.D > 0.0)


# --------------------------------------------------
# 7. Invalid inputs
# --------------------------------------------------

def test_nonpositive_depth_raises_error():
    """The model should reject non-positive fluid depth."""

    config, grid, model = setup_model()

    state = make_state(config, grid)
    state.D[0, 0] = 0.0

    with pytest.raises(ValueError):
        model.rhs(state)


def test_zero_epsilon_raises_error():
    """The model should reject epsilon equal to zero."""

    config, grid, model = setup_model(epsilon=0.0)

    state = make_state(config, grid)

    with pytest.raises(ValueError):
        model.rhs(state)

from operators import div, centred_grad_x, centred_grad_y

def test_periodic_divergence_has_zero_mean():
    config, grid, model = setup_model()

    rng = np.random.default_rng(42)
    u = rng.random((config.ny, config.nx))
    v = rng.random((config.ny, config.nx))

    divergence = div(u, v, config.dx, config.dy)

    assert np.isclose(np.sum(divergence), 0.0, atol=1e-10)


def test_periodic_gradient_has_zero_mean():
    config, grid, model = setup_model()

    rng = np.random.default_rng(42)
    q = rng.random((config.ny, config.nx))

    q_x = centred_grad_x(q, config.dx)
    q_y = centred_grad_y(q, config.dy)

    assert np.isclose(np.sum(q_x), 0.0, atol=1e-10)
    assert np.isclose(np.sum(q_y), 0.0, atol=1e-10)


def test_resting_fluid_over_variable_bathymetry():
    config, grid, model = setup_model()

    b = 0.1 * np.exp(
        -((grid.X - config.Lx/2)**2
          + (grid.Y - config.Ly/2)**2) / 0.05
    )

    model = Model(config, b)

    D = 1.0 + b
    gamma = np.ones((config.ny, config.nx))
    Q = D * gamma

    state = State(
        D=D,
        u=np.zeros((config.ny, config.nx)),
        v=np.zeros((config.ny, config.nx)),
        Q=Q
    )

    derivative = model.rhs(state)

    assert np.allclose(derivative.D, 0.0, atol=1e-10)
    assert np.allclose(derivative.u, 0.0, atol=1e-10)
    assert np.allclose(derivative.v, 0.0, atol=1e-10)
    assert np.allclose(derivative.Q, 0.0, atol=1e-10)

def test_buoyancy_advection():
    config, grid, model = setup_model()

    u0 = 0.2
    v0 = 0.1

    gamma = (
        1.0
        + 0.1 * np.sin(2 * np.pi * grid.X / config.Lx)
        + 0.1 * np.cos(2 * np.pi * grid.Y / config.Ly)
    )

    D = np.ones((config.ny, config.nx))
    Q = D * gamma

    state = State(
        D=D,
        u=np.full((config.ny, config.nx), u0),
        v=np.full((config.ny, config.nx), v0),
        Q=Q
    )

    derivative = model.rhs(state)

    Q_xface = cell_to_xface(Q)
    Q_yface = cell_to_yface(Q)

    flux_x = Q_xface * state.u
    flux_y = Q_yface * state.v

    expected = -div(
        flux_x,
        flux_y,
        config.dx,
        config.dy
    )

    assert np.allclose(
        derivative.Q,
        expected,
        atol=1e-10
    )