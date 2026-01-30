import numpy as np

G = 1.0
masses = np.array([1.0, 1.0, 1.0])

def total_energy(state):
    # Energia cinética
    kinetic = 0.0
    for i in range(3):
        v = state[i, 3:]
        kinetic += 0.5 * masses[i] * np.dot(v, v)

    # Energia potencial gravitacional
    potential = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r = np.linalg.norm(state[i, :3] - state[j, :3])
            potential -= G * masses[i] * masses[j] / r

    return kinetic + potential
