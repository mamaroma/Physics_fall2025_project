import numpy as np
from quantum_wave_packet.constants import HBAR

def gaussian_wave_packet(x, x0, k0, sigma):
    psi = np.exp(
        -(x - x0)**2 / (2 * sigma**2)
        + 1j * k0 * x
    )
    psi /= np.sqrt(np.trapz(np.abs(psi)**2, x))
    return psi