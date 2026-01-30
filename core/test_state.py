from core.state import State
from core.integrators import RK4Integrator
import numpy as np

state = State(
    positions=np.array([[-1,0,0],[1,0,0],[0,1,0]]),
    velocities=np.array([[0,0.3,0.2],[0,-0.3,-0.2],[0.3,0,0]])
)

rk4 = RK4Integrator(masses=[1,1,1], dt=0.01)

for _ in range(10):
    state = rk4.step(state)
    print(state.positions)
