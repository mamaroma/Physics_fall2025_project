import matplotlib.pyplot as plt
import numpy as np

def plot_wavefunction(x, psi, V, title=""):
    plt.figure(figsize=(8, 4))
    plt.plot(x, np.abs(psi)**2, label=r"$|\psi(x)|^2$")
    plt.plot(x, V / np.max(V + 1e-9), "--", label="V(x) (scaled)")
    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()