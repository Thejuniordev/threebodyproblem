import numpy as np

def total_energy(state):
    """
    state: (3, 6) -> [x, y, z, vx, vy, vz]
    """
    # Energia cinética
    kinetic = 0.0
    for body in state:
        vx, vy = body[3], body[4]
        kinetic += 0.5 * (vx**2 + vy**2)

    # Energia potencial
    potential = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r = np.linalg.norm(state[i, :2] - state[j, :2])
            potential -= 1.0 / r

    return kinetic + potential
