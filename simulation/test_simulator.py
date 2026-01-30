from core.state import State
from core.integrators import RK4Integrator
from simulation.simulator import Simulator
import numpy as np

state = State(
    positions=np.array([
        [-1, 0, 0],
        [ 1, 0, 0],
        [ 0, 1, 0]
    ]),
    velocities=np.array([
        [0, 0.3, 0.2],
        [0, -0.3, -0.2],
        [0.3, 0, 0]
    ])
)

rk4 = RK4Integrator(masses=[1,1,1], dt=0.01)
sim = Simulator(state, rk4)

sim.start()

for i in range(5):
    s = sim.update()
    print(f"t={sim.time:.2f}", s.positions)
