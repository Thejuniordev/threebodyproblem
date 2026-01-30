#gravitação, aceleração

import numpy as np

G = 1.0

def compute_accelerations(state, masses):
    n = len(masses)
    acc = np.zeros_like(state.positions)

    for i in range(n):
        for j in range(n):
            if i != j:
                diff = state.positions[j] - state.positions[i]
                dist = np.linalg.norm(diff) + 1e-5
                acc[i] += G * masses[j] * diff / dist**3

    return acc
