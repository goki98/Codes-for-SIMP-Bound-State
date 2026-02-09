import numpy as np
import matplotlib.pyplot as plt
from scipy.special import loggamma
import matplotlib.lines as mlines
from labellines import labelLines
g_x = 0.7
plt.rcParams.update({
    'font.size': 18,
    "text.usetex": True,
    'axes.titlesize': 18,
    'axes.labelsize': 21,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 18
})

def dot_product(a, b):
    return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]

def MatC(gx, p1, p2, q):
    denom1 = dot_product(p1, p1)
    denom2 = dot_product([p1[i] + p2[i] for i in range(4)],
                         [p1[i] + p2[i] for i in range(4)])
    denom3 = dot_product([p1[i] - q[i] for i in range(4)],
                         [p1[i] - q[i] for i in range(4)])
    V = np.zeros(4)
    for mu in range(4):
        term1 = 2 * p1[mu] / denom1
        term2 = 2 * (p1[mu] - p2[mu]) / denom2
        term3 = 2 * p2[mu] / denom3
        V[mu] = gx**3 * (term1 + term2 - term3)
    return V

def PertMatC(M):
    Mj = np.array(M[1:4])
    return np.dot(Mj, Mj)

# --- Velocity and mass setup ---
velocities = np.linspace(10, 10000, 80)
mx_values_fixed = [10, 150, 500]
n_states = [1, 2, 3]

colors = {10: 'r', 150: 'g', 500: 'b'}     # Color by mass
linestyles = {1: '-', 2: '--', 3: ':'}     # Line style by n

plt.figure(figsize=(7.3, 7.3))
x = 2.5

for mx in mx_values_fixed:
    for n in n_states:
        cs_curve = []
        for v_rel_km_s in velocities:
            beta_rel = v_rel_km_s / 2.998e5
            V0 = g_x**2
            mu = mx / 2
            a0 = 1 / (mu * V0)
            s = 4 * mx**2
            Ek = 0.5 * mu * v_rel_km_s**2

            # Binding energy (Coulomb)
            E_nc = (mu * V0**2) / (2 * n**2)
            MB_nc = 2 * mx - E_nc

            # Photon energy and momentum
            Wc = (s - MB_nc**2) / (2 * MB_nc)
            Qc = np.sqrt(Wc**2)

            # Wavefunction at r=0
            if n == 1:
                psi_n00 = 2 / (a0)**(3/2)
            elif n == 2:
                psi_n00 = 1 / np.sqrt(2*(a0**3))
            elif n == 3:
                psi_n00 = 2 / (3*np.sqrt(3*a0**3))

            # Scattering enhancement
            D = V0 / beta_rel
            PsiScatC = (2*np.pi*D)/(1 - np.exp(-2*np.pi*D))

            # Incoming 4-momenta
            p_mag = mx * beta_rel / (2 * np.sqrt(1 - beta_rel**2))
            E = np.sqrt(mx**2 + p_mag**2)
            p1 = [E, p_mag, 0, 0]
            p2 = [E, -p_mag, 0, 0]

            q_vec_nc = [Wc, 0, 0, Qc]

            Vnc = MatC(g_x, p1, p2, q_vec_nc)
            MPnc = PertMatC(Vnc)

            De_nc = Ek - E_nc
            Cn = 1/(16*np.pi*mx**2) * (De_nc) / (mu * v_rel_km_s**3)

            Mnc = 1/np.sqrt(2*mu) * psi_n00 * PsiScatC
            cs_curve.append(Cn * np.abs(Mnc)**2 * MPnc)

        plt.plot(velocities, cs_curve,
                 color=colors[mx],
                 linestyle=linestyles[n],
                 linewidth=x)
# Axis labels
plt.xlabel(r'$v_{\mathrm{rel}}$ ')
plt.ylabel(r'$\sigma_{\mathrm{BSF}}$')

# Shaded regions
plt.axvspan(0, 99, color='pink', alpha=0.3)
plt.axvspan(99, 999, color='yellow', alpha=0.3)
plt.axvspan(999, 10001, color='cyan', alpha=0.3)

# Log scale & grid
plt.yscale('log')
plt.xscale('log')
plt.xlim(10, 1e4)
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.grid(which="major", linestyle="--", linewidth=0.8, alpha=0.7)
plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

labelLines(
    plt.gca().get_lines(),
    xvals=[20, 80, 400, 20, 80, 400],  # one x-position per line
    align=True,
    fontsize=14
)


plt.tight_layout()
plt.text(0.07, 0.96, "(b)", transform=plt.gca().transAxes, fontsize=22)
plt.show()

