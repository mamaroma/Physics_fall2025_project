import numpy as np
from quantum_wave_packet.config import X_MIN, X_MAX, N_POINTS, DT, T_MAX

def create_spatial_grid():
    x = np.linspace(X_MIN, X_MAX, N_POINTS)
    dx = x[1] - x[0]
    return x, dx

def create_time_grid():
    t = np.arange(0, T_MAX, DT)
    return t