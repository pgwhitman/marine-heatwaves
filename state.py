from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    D: np.ndarray
    u: np.ndarray
    v: np.ndarray
    gamma: np.ndarray

    def copy(self):
        return State(
            D=self.D.copy(),
            u=self.u.copy(),
            v=self.v.copy(),
            gamma=self.gamma.copy()
        )

    def combine(self, other, a, b):
        """Return a*self + b*other."""
        return State(
            D=a*self.D + b*other.D,
            u=a*self.u + b*other.u,
            v=a*self.v + b*other.v,
            gamma=a*self.gamma + b*other.gamma
        )