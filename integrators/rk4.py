class RK4Integrator:
    def __init__(self, dt, derivative_fn):
        self.dt = dt
        self.f = derivative_fn

    def step(self, state):
        dt = self.dt
        k1 = self.f(state)
        k2 = self.f(state + 0.5 * dt * k1)
        k3 = self.f(state + 0.5 * dt * k2)
        k4 = self.f(state + dt * k3)
        return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
