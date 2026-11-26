import numpy as np


def grad_x(q, dx):
    """Gradient from cell centres to x-faces."""
    return (np.roll(q, -1, axis=1) - q) / dx


def grad_y(q, dy):
    """Gradient from cell centres to y-faces."""
    return (np.roll(q, -1, axis=0) - q) / dy


def div(u, v, dx, dy):
    """Divergence of face-centred velocity."""
    return (
        (u - np.roll(u, 1, axis=1)) / dx
        + (v - np.roll(v, 1, axis=0)) / dy
    )


def cell_to_xface(q):
    """Interpolate a cell-centred field to x-faces."""
    return 0.5 * (q + np.roll(q, -1, axis=1))


def cell_to_yface(q):
    """Interpolate a cell-centred field to y-faces."""
    return 0.5 * (q + np.roll(q, -1, axis=0))


def xface_to_cell(u):
    return 0.5 * (u + np.roll(u, 1, axis=1))


def yface_to_cell(v):
    return 0.5 * (v + np.roll(v, 1, axis=0))

def cell_to_corner(q):
    """Interpolate a cell-centred quantity to grid corners."""
    return 0.25 * (
        q
        + np.roll(q, -1, axis=1)
        + np.roll(q, -1, axis=0)
        + np.roll(np.roll(q, -1, axis=0), -1, axis=1)
    )


def corner_to_xface(q):
    """Interpolate a corner-centred quantity to x-faces."""
    return 0.5 * (q + np.roll(q, 1, axis=0))


def corner_to_yface(q):
    """Interpolate a corner-centred quantity to y-faces."""
    return 0.5 * (q + np.roll(q, 1, axis=1))


def centred_grad_x(q, dx):
    """Centred derivative at cell centres."""
    return (
        np.roll(q, -1, axis=1) - np.roll(q, 1, axis=1)
    ) / (2.0 * dx)


def centred_grad_y(q, dy):
    """Centred derivative at cell centres."""
    return (
        np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)
    ) / (2.0 * dy)