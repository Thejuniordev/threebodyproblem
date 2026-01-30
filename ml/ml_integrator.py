import torch
import torch.nn as nn
import numpy as np
from ml.normalization import normalize_state

class MLPredictor(nn.Module):
    def __init__(self, input_size=18, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        return self.net(x)


class MLIntegrator:
    def __init__(self, model, dt=0.01):
        self.model = model
        self.dt = dt

    def step(self, state):
        s_norm = normalize_state(state)
        inp = torch.tensor(s_norm.flatten(), dtype=torch.float32)

        with torch.no_grad():
            dstate_dt = self.model(inp).numpy()

        dstate_dt = dstate_dt.reshape(state.shape)
        return state + dstate_dt * self.dt
