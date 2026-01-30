class Simulator:
    def __init__(self, initial_state, integrator):
        self.initial_state = initial_state.copy()
        self.state = initial_state.copy()
        self.integrator = integrator
        self.history = []
        self.running = False
        self.time = 0.0

    def reset(self):
        self.state = self.initial_state.copy()
        self.time = 0.0
        self.history.clear()

    def start(self):
        self.running = True

    def pause(self):
        self.running = False

    def step(self):
        self.history.append(self.state.copy())
        self.state = self.integrator.step(self.state)
        return self.state

    def run(self, steps):
        for _ in range(steps):
            self.step()

    def update(self):
        """
        Usado em loop de tempo real
        """
        if self.running:
            return self.step()
        return self.state
