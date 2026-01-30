class VelocityVerlet:
    def __init__(self, dt, deriv_fn):
        self.dt = dt
        self.deriv_fn = deriv_fn

    def step(self, state):
        a0 = self.deriv_fn(state)[:, 3:]

        state_next = state.copy()
        state_next[:, :3] += (
            state[:, 3:] * self.dt
            + 0.5 * a0 * self.dt**2
        )

        a1 = self.deriv_fn(state_next)[:, 3:]

        state_next[:, 3:] += 0.5 * (a0 + a1) * self.dt
        return state_next
