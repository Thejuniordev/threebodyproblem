#RK4, euler, leapfrog

import numpy as np
from core.state import State
from core.physics import compute_accelerations

class RK4Integrator:
    def __init__(self, masses, dt):
        self.masses = masses
        self.dt = dt

    def step(self, state):
        def deriv(s):
            acc = compute_accelerations(s, self.masses)
            return np.hstack([s.velocities, acc]).flatten()

        y = state.as_vector()
        dt = self.dt

        k1 = deriv(state)
        k2 = deriv(State.from_vector(y + 0.5 * dt * k1))
        k3 = deriv(State.from_vector(y + 0.5 * dt * k2))
        k4 = deriv(State.from_vector(y + dt * k3))

        y_next = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return State.from_vector(y_next)
