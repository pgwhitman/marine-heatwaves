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
        Calculate the time derivatives of D, u, v and Q.

        D, Q: cell-centred
        u:    x-face-centred
        v:    y-face-centred
        """

        D = state.D
        u = state.u
        v = state.v
        Q = state.Q

        eps = self.epsilon
        dx = self.dx
        dy = self.dy

        if eps == 0.0:
            raise ValueError("epsilon must be non-zero.")

        if np.any(D <= 0.0):
            raise ValueError("Depth D must remain positive.")

        # Recover buoyancy from Q = D*gamma.
        gamma = Q / D

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
        # 2. Conservative buoyancy equation
        #       Q_t = - div(Q u)
        # --------------------------------------------------

        Q_xface = cell_to_xface(Q)
        Q_yface = cell_to_yface(Q)

        Q_flux_x = Q_xface * u
        Q_flux_y = Q_yface * v

        Q_t = -div(Q_flux_x, Q_flux_y, dx, dy)

        # --------------------------------------------------
        # 3. Relative vorticity at grid corners
        #       zeta = dv/dx - du/dy
        # --------------------------------------------------

        zeta = (
            (np.roll(v, -1, axis=1) - v) / dx
            - (np.roll(u, -1, axis=0) - u) / dy
        )

        # Absolute canonical vorticity.
        omega = eps * zeta + self.f0

        # Interpolate vorticity to velocity locations.
        omega_xface = corner_to_xface(omega)
        omega_yface = corner_to_yface(omega)

        # --------------------------------------------------
        # 4. Construct H and A at cell centres
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
            Q=Q_t,
        )