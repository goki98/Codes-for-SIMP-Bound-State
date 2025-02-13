#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:48:32 2024

@author: goki
"""

import numpy as np
from hepunits import GeV as m
from hepunits import MeV as m2
import matplotlib.pyplot as plt

# Parameters
rd = []
Yield = []
Xf = []

for X_f in np.arange(1, 1000, 0.001):  # Avoid X_f = 0 for numerical stability
    # Physical constants and parameters
    gs = 10.75
    g_dm = 1
    g_dm_s = g_dm**2 / np.sqrt(gs)
  
    
    # Equilibrium yield
    Y_eq = 0.145 * (g_dm / gs) * X_f**(3/2) * np.exp(-X_f)
    Xf.append(X_f)
    Yield.append(Y_eq)
    print(X_f, Y_eq)

# Plot setup
plt.figure(figsize=(10, 6))
plt.loglog(Xf, Yield, "r--", label=r"$Y_{\text{eq}}(x)$ (Equilibrium Yield)")

# Plot enhancements

plt.xlabel(r"$x = m_{\text{DM}} / T$", fontsize=12)
plt.ylabel(r"$Y(x)$", fontsize=12)
plt.ylim(1e-15, 1)
plt.title("Dark Matter Yield at equllibrium vs. $x$ (Log Scale)", fontsize=14)
plt.legend()
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
