import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class RealtimeView3D:
    def __init__(self, simulation):
        self.simulation = simulation

        # Figura
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(-5, 5)
        self.ax.set_title("Problema dos Três Corpos – Simulação em Tempo Real")

        # Pontos (corpos)
        colors = ["red", "blue", "green"]
        self.points = [
            self.ax.plot([], [], [], "o", color=c, markersize=8)[0]
            for c in colors
        ]

        # Trilhas (opcional, mas bonito)
        self.trails = [
            self.ax.plot([], [], [], color=c, alpha=0.6)[0]
            for c in colors
        ]
        self.trail_data = [[], [], []]

        # 🔑 Animação precisa ficar viva
        self.anim = FuncAnimation(
            self.fig,
            self.update,
            interval=30,
            blit=False,
            cache_frame_data=False
        )

    def update(self, frame):
        state = self.simulation.update()

        for i, point in enumerate(self.points):
            x, y, z = state[i, :3]

            # Atualiza ponto
            point.set_data([x], [y])
            point.set_3d_properties([z])

            # Atualiza trilha
            self.trail_data[i].append((x, y, z))
            xs, ys, zs = zip(*self.trail_data[i])
            self.trails[i].set_data(xs, ys)
            self.trails[i].set_3d_properties(zs)

        return self.points + self.trails

    def show(self):
        plt.show()
