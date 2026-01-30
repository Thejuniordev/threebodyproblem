#estrutura do

import numpy as np

class State:
    def __init__(self, positions, velocities):
        """
        positions: np.array (N, 3)
        velocities: np.array (N, 3)
        """
        self.positions = positions.astype(float)
        self.velocities = velocities.astype(float)

    def copy(self):
        return State(
            self.positions.copy(),
            self.velocities.copy()
        )

    def as_vector(self):
        """
        Retorna estado como vetor [x,y,z,vx,vy,vz,...]
        """
        return np.hstack([self.positions, self.velocities]).flatten()

    @staticmethod
    def from_vector(vector):
        data = vector.reshape(-1, 6)
        return State(data[:, :3], data[:, 3:])
