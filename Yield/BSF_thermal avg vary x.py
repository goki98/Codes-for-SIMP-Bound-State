#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 12:09:23 2025
@author: goki
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import loggamma
import matplotlib.pyplot as plt
import csv

# Constants
lam_x = 0.8
g_x = 0.1   # Coupling constant

# Global plot settings (if needed later)
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 21,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 18
})

# -------------------------------------------------------------------------
# 4-vector dot product (Minkowski metric: +, -, -, -)
def dot_product(a, b):
    return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]

# Coulomb matrix element
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

# Polarization-summed |M|^2
def PertMatC(M):
    Mj = np.array(M[1:4])
    return np.dot(Mj, Mj)

# Cross section computation
def CS(v_rel_km_s, mx):
    beta_rel = v_rel_km_s / 2.998e5
    V0 = g_x**2
    mu = mx / 2
    a0 = 1/(mu*V0)
    s = 4*mx**2
    Ek = 0.5 * mu * v_rel_km_s**2

    # Bound state energies
    E1c = (mu * V0**2) / 2
    E2c = (mu * V0**2) / (2 * 2**2)
    E3c = (mu * V0**2) / (2 * 3**2)

    # Energy differences
    De1c = abs(Ek - E1c)
    De2c = abs(Ek - E2c)
    De3c = abs(Ek - E3c)

    # Prefactors
    C1 = (1/(16*np.pi*mx**2)) * (De1c)/(mu*v_rel_km_s)
    C2 = (1/(16*np.pi*mx**2)) * (De2c)/(mu*v_rel_km_s)
    C3 = (1/(16*np.pi*mx**2)) * (De3c)/(mu*v_rel_km_s)

    # Wavefunction at r=0
    psi100c = 2 / a0**(3/2)
    psi200c = 1 / (2*a0**3)**0.5
    psi300c = 2 / (3*(3*a0**3)**0.5)

    D = V0 / beta_rel
    log_term = (D*np.pi/2) + loggamma(1 - 1j*D)
    PsiScatC = np.exp(log_term)

    # 4-vectors
    E = mx
    p_mag = mx * beta_rel / 2
    p1 = [E, p_mag, 0, 0]
    p2 = [E, -p_mag, 0, 0]

    # Photon momentum
    q1 = [De1c, 0, 0, De1c]
    q2 = [De2c, 0, 0, De2c]
    q3 = [De3c, 0, 0, De3c]

    # Matrix elements
    V1c = MatC(g_x, p1, p2, q1)
    V2c = MatC(g_x, p1, p2, q2)
    V3c = MatC(g_x, p1, p2, q3)

    MP1c = PertMatC(V1c)
    MP2c = PertMatC(V2c)
    MP3c = PertMatC(V3c)

    M1 = (1/np.sqrt(2*mu)) * psi100c * PsiScatC
    M2 = (1/np.sqrt(2*mu)) * psi200c * PsiScatC
    M3 = (1/np.sqrt(2*mu)) * psi300c * PsiScatC

    cs1c = C1 * np.abs(M1)**2 * MP1c
    cs2c = C2 * np.abs(M2)**2 * MP2c
    cs3c = C3 * np.abs(M3)**2 * MP3c

    return cs1c, cs2c, cs3c


# -------------------------------------------------------------------------
# Integrand for thermal average
def integrand(v_rel_km_s, x, mx):
    cs1, cs2, cs3 = CS(v_rel_km_s, mx)
    prefact = x**1.5 / (2*np.sqrt(np.pi))
    return prefact * v_rel_km_s**2 * np.exp(-0.25*x*v_rel_km_s**2) * (cs1 + cs2 + cs3)

# -------------------------------------------------------------------------
# Main loop over x values
if __name__ == "__main__":
    mx = 100  # GeV
    x_values = np.linspace(0.5, 100000, 200)  # avoid x=0 to prevent division by zero
    results = []

    for x in x_values:
        ThermCS, _ = quad(integrand, 0, np.inf, args=(x, mx))
        results.append((x, ThermCS))
        print(f"x = {x:.2f}, <σv²> = {ThermCS:.3e}")

    # Save results to CSV
    with open("Thermal_CS_vs_x_bsf.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "Thermal_CS"])
        writer.writerows(results)

    print("\nSaved to Thermal_CS_vs_x.csv")
