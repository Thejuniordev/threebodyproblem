import numpy as np
import torch
import matplotlib.pyplot as plt

from analysis.energy import total_energy
from pathlib import Path
from ml.ml_integrator import MLIntegrator, MLPredictor
from main import RK4Integrator

# -----------------------------
# PATHS (à prova de erro)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "ml" / "ml_model.pth"

# -----------------------------
# CONFIG
# -----------------------------
DT = 0.002
STEPS = 500

# -----------------------------
# ESTADO INICIAL
# -----------------------------
state0 = np.array([
    [-1.0,  0.0, 0.0,  0.0,  0.3,  0.2],
    [ 1.0,  0.0, 0.0,  0.0, -0.3, -0.2],
    [ 0.0,  1.0, 0.0,  0.3,  0.0,  0.0]
], dtype=np.float32)

# -----------------------------
# RK4
# -----------------------------
rk4 = RK4Integrator(DT)

# -----------------------------
# ML
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Modelo ML não encontrado em:\n{MODEL_PATH}\n"
        "Rode ml/train_ml.py primeiro."
    )

model = MLPredictor()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

ml = MLIntegrator(model)

# -----------------------------
# SIMULAÇÕES
# -----------------------------
rk_state = state0.copy()
ml_state = state0.copy()

rk_traj = []
ml_traj = []
errors = []
rk_energy = []
ml_energy = []

for _ in range(STEPS):
    rk_state = rk4.step(rk_state)
    ml_state = ml.step(ml_state)

    rk_traj.append(rk_state.copy())
    ml_traj.append(ml_state.copy())

    error = np.linalg.norm(rk_state - ml_state)
    errors.append(error)
    rk_energy.append(total_energy(rk_state))
    ml_energy.append(total_energy(ml_state))

rk_traj = np.array(rk_traj)
ml_traj = np.array(ml_traj)
errors = np.array(errors)

# -----------------------------
# PLOTS
# -----------------------------
plt.figure()
plt.plot(rk_energy, label="RK4")
plt.plot(ml_energy, label="ML")
plt.xlabel("Passo")
plt.ylabel("Energia Total")
plt.title("Conservação de Energia: RK4 vs ML")
plt.legend()
plt.show()

