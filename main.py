import os
import numpy as np
import torch

from simulation.simulator import Simulator
from visualization.realtime_view import RealtimeView3D
from ml.ml_integrator import MLIntegrator, MLPredictor
from ml.residual_integrator import ResidualMLIntegrator

# -----------------------------
# PARÂMETROS FÍSICOS
# -----------------------------
G = 1.0
masses = np.array([1.0, 1.0, 1.0])

# -----------------------------
# DINÂMICA DOS 3 CORPOS
# -----------------------------
def three_body_derivatives(state):
    deriv = np.zeros_like(state)

    for i in range(3):
        r_i = state[i, :3]
        v_i = state[i, 3:]
        a_i = np.zeros(3)

        for j in range(3):
            if i != j:
                diff = state[j, :3] - r_i
                dist = np.linalg.norm(diff) + 1e-5
                a_i += G * masses[j] * diff / dist**3

        deriv[i, :3] = v_i
        deriv[i, 3:] = a_i

    return deriv

# -----------------------------
# INTEGRADOR RK4 (fallback)
# -----------------------------
class RK4Integrator:
    def __init__(self, dt):
        self.dt = dt

    def step(self, state):
        k1 = three_body_derivatives(state)
        k2 = three_body_derivatives(state + 0.5 * self.dt * k1)
        k3 = three_body_derivatives(state + 0.5 * self.dt * k2)
        k4 = three_body_derivatives(state + self.dt * k3)
        return state + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# -----------------------------
# ESTADO INICIAL
# -----------------------------
initial_state = np.array([
    [-1.0,  0.0, 0.0,  0.0,  0.3,  0.2],
    [ 1.0,  0.0, 0.0,  0.0, -0.3, -0.2],
    [ 0.0,  1.0, 0.0,  0.3,  0.0,  0.0]
])

# -----------------------------
# MODELO ML
# -----------------------------
MODEL_PATH = "ml/ml_model.pth"

model = None
integrator = None

if os.path.exists(MODEL_PATH):
    model = MLPredictor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    integrator = MLIntegrator(model)
    print("✅ Usando integrador ML")
else:
    integrator = RK4Integrator(dt=0.01)
    print("⚠️ Modelo ML não encontrado, usando RK4")

# -----------------------------
# SIMULADOR
# -----------------------------
rk4 = RK4Integrator(dt=0.01)

residual_integrator = ResidualMLIntegrator(
    rk4_integrator=rk4,
    ml_model=model,
    device="cpu"
)

sim = Simulator(initial_state, integrator)
sim.start()

# -----------------------------
# VISUALIZAÇÃO
# -----------------------------
view = RealtimeView3D(sim)
view.show()
