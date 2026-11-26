from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Domain
    Lx: float = 1.0
    Ly: float = 1.0
    nx: int = 100
    ny: int = 100

    # Dimensionless model parameters
    epsilon: float = 0.1
    f0: float = 1.0

    # Time integration
    dt: float = 1.0e-4
    t_end: float = 0.5

    @property
    def dx(self):
        return self.Lx / self.nx

    @property
    def dy(self):
        return self.Ly / self.ny