import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from labellines import labelLines

# Constants
hbar_MeV_s = 6.582119569e-22

# Parameters
g = 0.7**2 / (4 * np.pi)
mx_values = np.linspace(1, 1000, 1000)
mu = mx_values / 2.0

Gamma_base = mu * g**5
ns = [1, 2, 3]
Gammas = {n: Gamma_base / (n**3) for n in ns}
taus = {n: hbar_MeV_s / Gammas[n] for n in ns}

plt.figure(figsize=(8,8))

for n in ns:
    plt.plot(mx_values, taus[n], label=f"n={n}")

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$m_\chi$ ')
plt.ylabel(r'$\tau$ ')
plt.grid(True, which='both', ls='--', alpha=0.6)

# ---- specify x positions for labels ----
label_x_positions = [50, 50, 50]   # one for each line (n=1, n=2, n=3)

labelLines(
    plt.gca().get_lines(),
    xvals=label_x_positions,
    align=True,
    fontsize=12
)
plt.xlim(1, 1000)
plt.tight_layout()
plt.text(0.07, 0.96, "(b)", transform=plt.gca().transAxes, fontsize=22)
plt.show()
