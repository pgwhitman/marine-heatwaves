import numpy as np

from state import State
from operators import (
    grad_x,
    grad_y,
    div,
    cell_to_xface,
    cell_to_yface,
    xface_to_cell,
    yface_to_cell,
    cell_to_corner,
    corner_to_xface,
    corner_to_yface,
    centred_grad_x,
    centred_grad_y,
)

class Model:
    def __init__(self, config, b):
        self.dx = config.dx
        self.dy = config.dy
        self.epsilon = config.epsilon
        self.f0 = config.f0

        # Fixed, cell-centred bathymetry.
        self.b = np.asarray(b, dtype=float)

    def rhs(self, state):
        """
        Calculate the time derivatives of D, u, v and gamma.

        D, gamma: cell-centred
        u:        x-face-centred
        v:        y-face-centred
        """

        D = state.D
        u = state.u
        v = state.v
        gamma = state.gamma

        eps = self.epsilon
        dx = self.dx
        dy = self.dy

        if eps == 0.0:
            raise ValueError("epsilon must be non-zero.")

        if np.any(D <= 0.0):
            raise ValueError("Depth D must remain positive.")

        # --------------------------------------------------
        # 1. Continuity equation
        #       D_t = - div(D u)
        # --------------------------------------------------

        D_xface = cell_to_xface(D)
        D_yface = cell_to_yface(D)

        flux_x = D_xface * u
        flux_y = D_yface * v

        D_t = -div(flux_x, flux_y, dx, dy)

        # --------------------------------------------------
        # 2. Buoyancy advection equation
        #       gamma_t = - u . grad(gamma)
        # --------------------------------------------------

        u_cell = xface_to_cell(u)
        v_cell = yface_to_cell(v)

        gamma_x = centred_grad_x(gamma, dx)
        gamma_y = centred_grad_y(gamma, dy)

        gamma_t = -(u_cell * gamma_x + v_cell * gamma_y)

        # --------------------------------------------------
        # 3. Relative vorticity at grid corners
        #
        #       zeta = dv/dx - du/dy
        # --------------------------------------------------

        zeta = (
            (np.roll(v, -1, axis=1) - v) / dx
            - (np.roll(u, -1, axis=0) - u) / dy
        )

        # Absolute canonical vorticity:
        #       curl(v_c) = (epsilon*zeta + f0) k
        omega = eps * zeta + self.f0

        # Interpolate vorticity to the velocity locations.
        omega_xface = corner_to_xface(omega)
        omega_yface = corner_to_yface(omega)

        # --------------------------------------------------
        # 4. Construct H and A at cell centres
        #
        # H = epsilon/2 |u|^2 + (D-b)*gamma^2
        # A = (D-b)^2 / (2D)
        # --------------------------------------------------

        u_cell = xface_to_cell(u)
        v_cell = yface_to_cell(v)

        speed_squared = u_cell**2 + v_cell**2

        H = (
            0.5 * eps * speed_squared
            + (D - self.b) * gamma**2
        )

        A = (D - self.b)**2 / (2.0 * D)

        # --------------------------------------------------
        # 5. Momentum equation: x-component
        # --------------------------------------------------

        # Interpolate y-velocity to x-faces.
        v_xface = cell_to_xface(yface_to_cell(v))

        H_x = grad_x(H, dx)
        gamma_squared_x = grad_x(gamma**2, dx)

        A_xface = cell_to_xface(A)

        u_t = (
            v_xface * omega_xface
            - H_x
            + A_xface * gamma_squared_x
        ) / eps

        # --------------------------------------------------
        # 6. Momentum equation: y-component
        # --------------------------------------------------

        # Interpolate x-velocity to y-faces.
        u_yface = cell_to_yface(xface_to_cell(u))

        H_y = grad_y(H, dy)
        gamma_squared_y = grad_y(gamma**2, dy)

        A_yface = cell_to_yface(A)

        v_t = (
            -u_yface * omega_yface
            - H_y
            + A_yface * gamma_squared_y
        ) / eps

        return State(
            D=D_t,
            u=u_t,
            v=v_t,
            gamma=gamma_t,
        )
