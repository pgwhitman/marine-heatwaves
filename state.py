from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    D: np.ndarray
    u: np.ndarray
    v: np.ndarray
    Q: np.ndarray

    def copy(self):
        return State(
            D=self.D.copy(),
            u=self.u.copy(),
            v=self.v.copy(),
            Q=self.Q.copy()
        )

    def combine(self, other, a, b):
        """Return a*self + b*other."""
        return State(
            D=a*self.D + b*other.D,
            u=a*self.u + b*other.u,
            v=a*self.v + b*other.v,
            Q=a*self.Q + b*other.Q
        )