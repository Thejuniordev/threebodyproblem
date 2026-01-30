import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from ml.ml_integrator import MLPredictor
from physics.energy_torch import total_energy
from physics.physics_loss import energy_loss
from main import RK4Integrator
from pathlib import Path

# --------------------------------
# CONFIGURAÇÕES
# --------------------------------
DT = 0.01
EPOCHS = 200
STEPS_PER_TRAJ = 200
TRAJECTORIES = 50
LR = 1e-3

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "ml" / "ml_model.pth"
PHYS_WEIGHT = 0.1

# --------------------------------
# ESTADO INICIAL BASE
# --------------------------------
BASE_STATE = np.array([
    [-1.0,  0.0, 0.0,  0.0,  0.3,  0.2],
    [ 1.0,  0.0, 0.0,  0.0, -0.3, -0.2],
    [ 0.0,  1.0, 0.0,  0.3,  0.0,  0.0]
], dtype=np.float32)

# --------------------------------
# DATASET (RK4)
# --------------------------------
def generate_dataset():
    rk4 = RK4Integrator(DT)

    X = []
    y = []

    for _ in range(TRAJECTORIES):
        state = BASE_STATE + 0.05 * np.random.randn(*BASE_STATE.shape)

        for _ in range(STEPS_PER_TRAJ):
            next_state = rk4.step(state)

            X.append(state.flatten())
            rk_next = rk4.step(state)
            y.append((next_state - rk_next).flatten())

            state = next_state

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y

# --------------------------------
# TREINAMENTO
# --------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Usando dispositivo: {device}")

    print("🔧 Gerando dataset com RK4...")
    X_np, y_np = generate_dataset()

    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    model = MLPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("🚀 Iniciando treinamento...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        pred = model(X)

        L_DATA = criterion(pred, y)
        L_PHYS = energy_loss(X, pred, DT, total_energy)

        loss = L_DATA + PHYS_WEIGHT * L_PHYS

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.6f} | "
                f"Data: {L_DATA.item():.6f} | "
                f"Phys: {L_PHYS.item():.6f}"
            )

    # --------------------------------
    # SALVAR MODELO
    # --------------------------------
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Modelo salvo em {MODEL_PATH}")

# --------------------------------
# MAIN
# --------------------------------
if __name__ == "__main__":
    train()
