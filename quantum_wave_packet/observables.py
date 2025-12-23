import numpy as np

def norm(psi, x):
    return np.trapz(np.abs(psi)**2, x)

def expectation_x(psi, x):
    return np.trapz(np.conj(psi) * x * psi, x).real

def variance_x(psi, x):
    x_mean = expectation_x(psi, x)
    return np.trapz(np.conj(psi) * (x - x_mean)**2 * psi, x).real

def tunneling_probability(psi, x, barrier_position):
    mask = x > barrier_position
    return np.trapz(np.abs(psi[mask])**2, x[mask])