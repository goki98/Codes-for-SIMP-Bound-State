import numpy as np
from scipy.integrate import quad
from scipy.special import loggamma
import matplotlib.pyplot as plt
from labellines import labelLines
import matplotlib.lines as mlines

# Constants
mh = 125000  # MeV (mass of Higgs boson)
mp = 0.05    # MeV (mass of mediator)
lam_xh = 1e-3
lam_hp = 0.3
lam_x = 0.8
g_x = 0.7

plt.rcParams.update({
    'font.size': 18,          # Base font size
    "text.usetex": True,      # Latex switch
    'axes.titlesize': 18,     # Subplot titles
    'axes.labelsize': 21,     # Axis labels
    'xtick.labelsize': 18,    # X-axis ticks
    'ytick.labelsize': 18,    # Y-axis ticks
    'legend.fontsize': 18,    # Legend text
    'figure.titlesize': 18    # Main title
})

# Particle masses
M_ele = 0.511
M_mu  = 105.658
M_Pc  = 139.57

# --- Functions ---
def PetMatAnn(mf):
    """Matrix element for DM annihilation into fermion f."""
    s = 4 * mx**2
    lam_hf = mf
    M1 = 2 * (s - 4 * mf**2)
    M2 = ((lam_xh * lam_hf)**2 / (s - mh**2))**2
    return M1 * M2

def AnnDecCS(mf, MatSq):
    Prefact2 = 1 / (64 * mx**2 * np.pi)
    CS = Prefact2 * np.sqrt(4 * mx**2 - 4 * mf**2) * MatSq
    return CS

def pertAnnPion(mp):
    s = 4 * mx**2
    lam_hp = mp**2
    pre = lam_xh * lam_hp / 2
    M_sq = (pre / (s - mh**2))**2
    return M_sq

def AnnDecCS_p(mp, MatSq):
    Prefact2 = 1 / (64 * mx**2 * np.pi)
    CS = Prefact2 * np.sqrt(4 * mx**2 - 4 * mp**2) * MatSq
    return CS

def PsiScatC(mx, v_rel_km_s):
    """Sommerfeld-like scattering state factor."""
    beta_rel = v_rel_km_s / 2.998e5
    mu = mx / 2
    V0 = g_x**2
    D = V0 / beta_rel
    log_term = (D * np.pi / 2) + loggamma(1 - 1j * D)
    return np.exp(log_term)

# --- Velocity and mass setup ---
velocities = np.linspace(10, 10000, 800)
mx_values_fixed = [500, 150, 10]

# Colors and linestyles
colors = {500: 'b', 150: 'g', 10: 'r'}
linestyles = {'e': '-', 'mu': '--', 'pi': ':'}

plt.figure(figsize=(8, 8))

# --- Main Loop ---
for mx in mx_values_fixed:
    for final_state in ['e', 'mu', 'pi']:
        cs_curve = []
        for v_rel_km_s in velocities:
            psi_scat_sq = np.abs(PsiScatC(mx, v_rel_km_s))**2

            if final_state == 'e':
                matsq = PetMatAnn(M_ele) * psi_scat_sq
                cs_curve.append(AnnDecCS(M_ele, matsq))
            elif final_state == 'mu':
                matsq = PetMatAnn(M_mu) * psi_scat_sq
                cs_curve.append(AnnDecCS(M_mu, matsq))
            elif final_state == 'pi':
                matsq = PetMatAnn(M_Pc) * psi_scat_sq
                cs_curve.append(AnnDecCS_p(M_Pc, matsq))

        plt.plot(
            velocities, cs_curve,
            color=colors[mx],
            linestyle=linestyles[final_state],
            linewidth=2.5
        )

# --- Plot labels ---
plt.xlabel(r'$v_{\mathrm{rel}}$')
plt.ylabel(r'$\sigma_{\mathrm{Ann}}$')

# --- Shaded regions for astrophysical environments ---
plt.axvspan(0, 99, color='pink', alpha=0.3, label='Dwarf Galaxy')
plt.axvspan(99, 999, color='yellow', alpha=0.3, label='Galactic Center')
plt.axvspan(999, 10001, color='cyan', alpha=0.3, label='Galaxy Cluster')

plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.xlim(10, 1e4)
plt.yscale('log')
plt.xscale('log')
plt.grid(which="major", linestyle="--", linewidth=0.8, alpha=0.7)
plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

# --- Custom 6-entry legend ---
color_handles = [
    mlines.Line2D([], [], color='b', linestyle='-', linewidth=2.5, label=r'$m_\chi=500$ MeV'),
    mlines.Line2D([], [], color='g', linestyle='-', linewidth=2.5, label=r'$m_\chi=150$ MeV'),
    mlines.Line2D([], [], color='r', linestyle='-', linewidth=2.5, label=r'$m_\chi=10$ MeV')
]

style_handles = [
    mlines.Line2D([], [], color='k', linestyle='-', linewidth=2.5, label=r'$e$'),
    mlines.Line2D([], [], color='k', linestyle='--', linewidth=2.5, label=r'$\mu$'),
    mlines.Line2D([], [], color='k', linestyle=':', linewidth=2.5, label=r'$\pi$')
]

handles = color_handles + style_handles


plt.text(0.07, 0.96, "(b)", transform=plt.gca().transAxes, fontsize=22)
plt.show()
