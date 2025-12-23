import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from quantum_wave_packet.constants import HBAR, MASS

def crank_nicolson_step(psi, V, dx, dt):
    N = len(psi)

    alpha = 1j * HBAR * dt / (2 * MASS * dx**2)

    main_diag = 1 + 2 * alpha + 1j * dt * V / (2 * HBAR)
    off_diag = -alpha * np.ones(N - 1)

    A = diags(
        [off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        format="csc"
    )

    main_diag_B = 1 - 2 * alpha - 1j * dt * V / (2 * HBAR)
    B = diags(
        [-off_diag, main_diag_B, -off_diag],
        offsets=[-1, 0, 1],
        format="csc"
    )

    rhs = B @ psi
    psi_next = splu(A).solve(rhs)

    return psi_next