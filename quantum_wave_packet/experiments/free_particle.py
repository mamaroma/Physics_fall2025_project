import numpy as np

from quantum_wave_packet.grid import create_spatial_grid, create_time_grid
from quantum_wave_packet.initial_state import gaussian_wave_packet
from quantum_wave_packet.schrodinger_solver import crank_nicolson_step
from quantum_wave_packet.visualization import plot_wavefunction
from quantum_wave_packet.animation import animate_wave_packet
from quantum_wave_packet.config import X0, K0, SIGMA, DT
from quantum_wave_packet.potentials import free_particle

def run():
    x, dx = create_spatial_grid()
    t = create_time_grid()

    V = free_particle(x)
    psi = gaussian_wave_packet(x, X0, K0, SIGMA)

    psi_t = [psi.copy()]

    for _ in t[1:]:
        psi = crank_nicolson_step(psi, V, dx, DT)
        psi_t.append(psi.copy())

    plot_wavefunction(x, psi_t[-1], V, "Free particle: wave packet dispersion")
    animate_wave_packet(x, psi_t, V)

if __name__ == "__main__":
    run()