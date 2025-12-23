import numpy as np
from quantum_wave_packet.constants import (
    OMEGA,
    BARRIER_HEIGHT,
    BARRIER_WIDTH,
    WELL_DEPTH,
    WELL_WIDTH
)

def free_particle(x):
    return np.zeros_like(x)

def harmonic_oscillator(x):
    return 0.5 * OMEGA**2 * x**2

def potential_barrier(x):
    V = np.zeros_like(x)
    mask = np.abs(x) < BARRIER_WIDTH / 2
    V[mask] = BARRIER_HEIGHT
    return V

def finite_well(x):
    V = np.zeros_like(x)
    mask = np.abs(x) < WELL_WIDTH / 2
    V[~mask] = WELL_DEPTH
    return V