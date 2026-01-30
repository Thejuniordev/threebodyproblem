import torch
import numpy as np

class ResidualMLIntegrator:
    def __init__(self, rk4_integrator, ml_model, device="cpu"):
        self.rk4 = rk4_integrator
        self.model = ml_model
        self.device = device

    def step(self, state):
        # Passo base (física clássica)
        rk_next = self.rk4.step(state)

        # Residual aprendido
        with torch.no_grad():
            x = torch.tensor(
                state.flatten(),
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            residual = self.model(x).squeeze(0).cpu().numpy()
            residual = residual.reshape(3, 6)

        return rk_next + residual
