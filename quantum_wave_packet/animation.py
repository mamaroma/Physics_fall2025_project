import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_wave_packet(x, psi_t, V):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    pot, = ax.plot(x, V / np.max(V + 1e-9), "--")

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, np.max(np.abs(psi_t[0])**2) * 1.5)

    def update(frame):
        line.set_data(x, np.abs(psi_t[frame])**2)
        return line,

    ani = FuncAnimation(fig, update, frames=len(psi_t), interval=50)
    plt.show()