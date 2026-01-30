import torch

G = 1.0

def total_energy(state):
    """
    state: Tensor [B, 18]
    Retorna: Tensor [B]
    """
    device = state.device
    masses = torch.tensor([1.0, 1.0, 1.0], device=device)

    # reshape -> [B, 3, 6]
    state = state.view(-1, 3, 6)

    # --------------------
    # Energia cinética
    # --------------------
    v = state[:, :, 3:]                 # [B, 3, 3]
    kinetic = 0.5 * masses * (v ** 2).sum(dim=2)
    kinetic = kinetic.sum(dim=1)        # [B]

    # --------------------
    # Energia potencial
    # --------------------
    potential = torch.zeros_like(kinetic)

    for i in range(3):
        for j in range(i + 1, 3):
            r = torch.norm(
                state[:, i, :3] - state[:, j, :3],
                dim=1
            )
            potential -= G * masses[i] * masses[j] / r

    return kinetic + potential
