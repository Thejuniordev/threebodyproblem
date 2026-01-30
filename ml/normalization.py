import numpy as np

# Limites esperados (ajustáveis depois)
POS_SCALE = 2.0
VEL_SCALE = 1.0

def normalize_state(state):
    s = state.copy()
    s[:, :3] /= POS_SCALE
    s[:, 3:] /= VEL_SCALE
    return s

def denormalize_state(state):
    s = state.copy()
    s[:, :3] *= POS_SCALE
    s[:, 3:] *= VEL_SCALE
    return s
