import numpy as np

def default_three_body_state():
    return np.array([
        [-1.0,  0.0, 0.0,  0.0,  0.3,  0.2],
        [ 1.0,  0.0, 0.0,  0.0, -0.3, -0.2],
        [ 0.0,  1.0, 0.0,  0.3,  0.0,  0.0]
    ])
