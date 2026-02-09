# -*- coding: utf-8 -*-
"""
Combined Boltzmann Evolution Solver
-----------------------------------
Includes:
1. Coupled two-component Boltzmann system (Yχ, YB)
2. Single Boltzmann equation (self-annihilation term)
Plots both evolutions and observed relic density line.

Author: Gokhula Prasad & Prasanna
Date: Oct 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from labellines import labelLines

# ============================================================
# ============= COMMON CONSTANTS AND FUNCTIONS ===============
# ============================================================
gs = 10.75
g_dm = 1.0
MPl = 1.22e22  # MeV
mx = 100       # MeV
epi = 1j * 1e-10

def y_eq_X(x): 
    return 0.145 * (g_dm / gs) * x**1.5 * np.exp(-x)

def y_eq_B(x): 
    return 0.145 * (g_dm / gs) * 2 * x**1.5 * np.exp(-2*x)

def y_eq_DP(x): 
    return 0.278 * (g_dm / gs)

def entropy(x): 
    return (2 * np.pi**2) / 45 * gs * (mx / x)**3


# ============================================================
# =========== MODEL 1: COUPLED BOLTZMANN SYSTEM ==============
# ============================================================

def dydx_coupled(x, y, Tcs_tuple):
    yX, yB = y
    (Tcs1, Tcs2, Tcs3, Tcs4, Tcs5, Tcs6,
     Tcs7, Tcs8, Tcs9, Tcs10, Tcs11, Tcs12) = Tcs_tuple
    
    eqX, eqB, eqDP = y_eq_X(x), y_eq_B(x), y_eq_DP(x)
    yDP = eqDP
    s = entropy(x)
    Hm = 1.67 * np.sqrt(gs) * mx**2 / MPl

    # --- Cutoff for thermal averaging ---
    cutoff = 300
    if x > cutoff:
        tcs1 = tcs2 = tcs4 = tcs5 = tcs6 = tcs9 = tcs10 = tcs11 = tcs12 = 0.0
    else:
        tcs1 = Tcs1 / (eqB * eqDP)
        tcs2 = Tcs2 / (eqX**2)
        tcs4 = Tcs4 / (eqB * eqDP)
        tcs5 = Tcs5 / (eqDP**2)
        tcs9 = Tcs9 
        tcs11 = Tcs11 / (eqDP)
        tcs12 = Tcs12 / (eqDP)
        tcs6 = Tcs6 / (eqX**2)
        tcs10 = Tcs10 / (eqX**2)

    # --- Reaction terms ---
    t1 = s**2 * tcs1 * (yX**2 * eqDP * eqB - eqX**2 * yDP * yB)
    t2 = s**2 * tcs2 * (yDP * yB * eqX**2 - eqDP * eqB * yX**2)
    t3 = s**4 * Tcs3 * (yX**4 - eqX**2 * yX**2)
    t4 = s**2 * tcs4 * (yX**4 * eqDP * eqB - eqX**4 * yDP * yB)
    t5 = s**2 * tcs5 * (yX**2 * eqDP**2 - eqX**2 * yDP**2)
    t6 = s**2 * Tcs6 * (yDP**2 * eqX**2 - eqDP**2 * yX**2)
    t9 = s**3 * Tcs9 * (yX**2 * yB - eqB**2 * yX**2)
    t10 = s**2 * tcs10 * (yB**2 * eqX**2 - eqB**2 * yX**2)
    t11 = s**3 * tcs11 * (yX**2 * yB * eqDP - eqX**2 * yDP * yB)
    t12 = s**2 * tcs12 * (yB**2 * eqDP - eqB * yB * yDP)

    Prefactor = x / (s * Hm)
    dy1 = Prefactor * (t2 - t1 - t3 - t4 + t10 + t9 - t11)
    dy2 = Prefactor * (t1 - t2 + t4 - t10 - t9 - t12)
    return np.array([dy1, dy2])


# ============================================================
# ============ MODEL 2: SINGLE EQUATION SYSTEM ===============
# ============================================================

def dydx_single(x, y, Tcs_tuple):
    yX = y[0]
    Tcs3 = Tcs_tuple[0]
    eqX = y_eq_X(x)
    s = entropy(x)
    t3 = Tcs3 * (yX**4 - eqX**2 * yX**2)
    Prefactor = -0.508 * (gs**3 / np.sqrt(gs)) * MPl * (mx**7 / x**8)
    dy1 = Prefactor * t3
    return np.array([dy1])


# ============================================================
# ================= NUMERICAL SOLVER ==========================
# ============================================================

def backward_euler(f, x0, y0, x_end, h, Tcs_tuple):
    x_values = [x0]
    y_values = [y0]
    x = x0
    y = np.array(y0)

    while x < x_end:
        def equation(y_new):
            return y_new - y - h * f(x + h, y_new, Tcs_tuple)
        
        y_new = fsolve(equation, y)
        x += h
        x_values.append(x)
        y_values.append(y_new)
        y = y_new
    return np.array(x_values), np.array(y_values)


# ============================================================
# ===================== RUN MODEL 1 ===========================
# ============================================================

Tcs_tuple_full = (
    2.5e-7, 0.5e-26, 5.230316e-7, 2.083544e-05, 0, 0,
    2.5e-10, 9.5e-9, 7.365470e-20, 1.071169e-4, 5.030030e-1, 5.692892e-1
)

x0, x_end, h = 1.0, 1000.0, 0.5
y0 = [0.02, 0.01]

x_vals1, y_vals1 = backward_euler(dydx_coupled, x0, y0, x_end, h, Tcs_tuple_full)
y_vals1 = np.array(y_vals1)

# ============================================================
# ===================== RUN MODEL 2 ===========================
# ============================================================

Tcs_tuple_simple = (5.230316e-7,)
y0_simple = [0.1]

x_vals2, y_vals2 = backward_euler(dydx_single, x0, y0_simple, x_end, h, Tcs_tuple_simple)
y_vals2 = np.array(y_vals2)


# ============================================================
# ======================= PLOTTING ============================
# ============================================================

plt.figure(figsize=(10, 8))

# --- Model 1 ---
plt.loglog(x_vals1, y_vals1[:, 0], 'r', label=r"$Y_\chi$ ")
plt.loglog(x_vals1, y_vals1[:, 1], 'g', label=r"$Y_B$")

# --- Model 2 ---
plt.loglog(x_vals2, y_vals2[:, 0], 'b', label=r"$Y_{free}$")

# --- Equilibrium yields ---
Xf = np.arange(1, 1000, 1.0)
plt.loglog(Xf, [y_eq_X(x) for x in Xf], "r--")
plt.loglog(Xf, [y_eq_B(x) for x in Xf], "g--")

# --- Observed relic density line ---
Omega_h2_obs = 0.12
conv_factor = 2.752e5 * 1e2  # from your code
Y_obs = Omega_h2_obs / conv_factor

plt.axhline(Y_obs, color='purple', linestyle=':', linewidth=2.2)
plt.text(3, Y_obs*1.3, r" $(\Omega h^2 = 0.12)$",
         color='purple', fontsize=12)
# --- Vertical lines for key x-values ---
plt.axvline(x=17, color='blue', linestyle=':', linewidth=1.8, label=r"$x_f \approx 15$")
plt.axvline(x=19, color='red', linestyle=':', linewidth=1.8, label=r"$x_f \approx 20$")
plt.axvline(x=300, color='green', linestyle=':', linewidth=1.8, label=r"$x_c = 300$")

# --- Plot formatting ---
plt.ylim(1e-20, 1e0)
plt.xlim(0.25e1, 1e3)
plt.xlabel(r'$x = M_\chi / T$')
plt.ylabel( '$Y$')
plt.title(
    r'$m_\chi = 150\,\mathrm{MeV},\; g_\chi = 0.7,\; \lambda_\chi = 0.8$',
    y=1.02
)


plt.grid(True, which='both', ls='--', alpha=0.6)
#plt.legend()
labelLines(plt.gca().get_lines(), align=True, fontsize=12, xvals=[50, 200, 200])
plt.show()

# ============================================================
# ====================== FINAL OUTPUTS ========================
# ============================================================

print("=== Final Yields (Coupled System) ===")
print(f"Yχ = {y_vals1[-1, 0]:.3e}")
print(f"YB = {y_vals1[-1, 1]:.3e}")

RD = 2.752e5 * 1e2 * (y_vals1[-1, 0] + y_vals1[-1, 1])
print(f"Computed Relic Density Ωh² = {RD:.3e}")

print("\n=== Final Yield (Single Equation) ===")
print(f"Y = {y_vals2[-1, 0]:.3e}")
