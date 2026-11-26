import numpy as np


class Grid:
    def __init__(self, config):
        self.nx = config.nx
        self.ny = config.ny

        self.dx = config.dx
        self.dy = config.dy

        # Cell centres
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.y = (np.arange(self.ny) + 0.5) * self.dy

        self.X, self.Y = np.meshgrid(self.x, self.y)

        # All fields have shape (ny, nx).
        # D, gamma: cell-centred
        # u: x-face-centred
        # v: y-face-centred