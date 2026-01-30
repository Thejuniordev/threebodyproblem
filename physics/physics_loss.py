import torch

def energy_loss(state, dstate_dt, dt, energy_fn):
    next_state = state + dstate_dt * dt

    E0 = energy_fn(state)
    E1 = energy_fn(next_state)

    return torch.mean((E1 - E0) ** 2)

