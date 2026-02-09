#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 12:09:23 2025

@author: goki
"""

import numpy as np
from scipy.integrate import quad
from labellines import labelLines
import matplotlib.pyplot as plt

# Constants

mp=0.08




lam_x = 0.8
g_x = 0.7  # Coupling constant
# global plot text size 
plt.rcParams.update({
    "text.usetex": True,
    'font.size': 18,          # Base font size
    'axes.titlesize': 18,     # Subplot titles
    'axes.labelsize': 21,     # Axis labels
    'xtick.labelsize': 18,    # X-axis ticks
    'ytick.labelsize': 18,    # Y-axis ticks
    'legend.fontsize': 16,    # Legend text
    'figure.titlesize': 18    # Main title
})

# Coulomb wavefunction
# Explicit Associated Laguerre Polynomial for n=0,1,2,3
def explicit_assoc_laguerre(n, alpha, x):
    """Compute associated Laguerre polynomial L_n^{(alpha)}(x) for n=0,1,2,3"""
    if n == 0:
        return 1.0
    elif n == 1:
        return -x + alpha + 1
    elif n == 2:
        return 0.5 * (x**2 - 2*(alpha+2)*x + (alpha+1)*(alpha+2))
    elif n == 3:
        term1 = -x**3
        term2 = 3*(alpha + 3)*x**2
        term3 = -3*(alpha+2)*(alpha+3)*x
        term4 = (alpha+1)*(alpha+2)*(alpha+3)
        return (term1 + term2 + term3 + term4) / 6.0
    else:
        raise ValueError("n must be 0, 1, 2, or 3")

# Explicit Jacobi Polynomial for n=0,1,2,3
def explicit_jacobi(n, a, b, x):
    """Compute Jacobi polynomial P_n^{(a,b)}(x) for n=0,1,2,3"""
    if n == 0:
        return 1.0
    elif n == 1:
        return 0.5 * (a - b + (a + b + 2)*x)
    elif n == 2:
        term1 = (a+b+3)*(a+b+4)*x**2
        term2 = 2*(a - b)*(a + b + 3)*x
        term3 = (a - b)**2 - (a + b + 4)
        return (term1 + term2 + term3) / 8.0
    elif n == 3:
        # Compute P1 and P2 explicitly
        P1 = 0.5 * (a - b + (a + b + 2)*x)
        term1 = (a+b+3)*(a+b+4)*x**2
        term2 = 2*(a - b)*(a + b + 3)*x
        term3 = (a - b)**2 - (a + b + 4)
        P2 = (term1 + term2 + term3) / 8.0
        
        # Use recurrence relation for P3
        A = a+b+5
        B = (a+b+4)*(a+b+6)*x + (a**2 - b**2)
        C = 2*(a+2)*(b+2)*(a+b+6)
        denom = 6 * (a+b+3) * (a+b+4)
        return (A * B * P2 - C * P1) / denom
    else:
        raise ValueError("n must be 0, 1, 2, or 3")

# Coulomb wavefunction (n=1,2,3; l=0)
def get_normalized_wavefunction_C(n, l, m):
    """Compute normalized Coulomb wavefunction for given quantum numbers"""
    # Binding energy for Coulomb potential
    Enlm = -(mu * V0**2) / (2 * n**2)
    a = -2 * mu * Enlm  # = (mu * V0_C / n)**2
    kappa = np.sqrt(a)
    
    def unnormalized_psi(r):
        # Quantum number for Laguerre: k = n - l - 1
        k = n - l - 1
        return r**l * np.exp(-kappa * r) * explicit_assoc_laguerre(k, 2*l+1, 2*kappa*r)
    
    # Normalization
    def integrand(r):
        return np.abs(unnormalized_psi(r))**2 * r**2
    
    radial_integral, _ = quad(integrand, 0, np.inf)
    total_integral = 4 * np.pi * radial_integral
    A = 1 / np.sqrt(total_integral) if total_integral != 0 else 1.0
    
    def normalized_psi(r):
        return A * unnormalized_psi(r)
   
    return normalized_psi, A

# Hulthen wavefunction (n=1,2,3; l=0)
def get_normalized_wavefunction_H(n, l, m):
    """Compute normalized Hulthen wavefunction for given quantum numbers"""
    q = 1  # Deformation parameter
    # Binding energy for Hulthen potential
    Enlm = -1/(2*mu) * ((V0 * mu)/(n*q) - (n*mp)/2)**2
    alpha = (-2 * mu * Enlm) / (mp**2)  # Dimensionless parameter
    
    def unnormalized_psi(r):
        z = 1 - 2 * np.exp(-mp * r)
        prefactor = np.exp(-r * mp * np.sqrt(alpha)) * (1 - q * np.exp(-r * mp))**(l + 1) / r
        jac_arg = explicit_jacobi(n, 2*np.sqrt(alpha), 2*l+1, z)
        return prefactor * jac_arg
    
    # Normalization
    def integrand(r):
        return np.abs(unnormalized_psi(r))**2 * r**2
    
    radial_integral, _ = quad(integrand, 0, np.inf)
    total_integral = 4 * np.pi * radial_integral
    A = 1 / np.sqrt(total_integral) if total_integral != 0 else 1.0
    
    def normalized_psi(r):
        return A * unnormalized_psi(r)
    
    return normalized_psi, A, alpha
#Four vector dot product
def dot_product(a, b):
    """Minkowski dot product for four-vectors (+,-,-,-)"""
    return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
# Perturbative amplitudes
#matrix element in four vector form
#columb
#columb
def MatC(gx, mp, p1, p2, q):
   # Denominator terms
   denom1 = dot_product(p1, p1)
   denom2 = dot_product(q, q)- mx**2
   denom3 =  dot_product([p1[i] - q[i] for i in range(4)], [p1[i] - q[i] for i in range(4)])
   Denom=denom1*denom2*denom3
   
   V = np.zeros(4)
   for mu in range(4):
       Num1=2*dot_product(p1, p1)*p2[mu]*(mx**2-dot_product(q, q))
       Num2=2*p1[mu]*dot_product([p1[i] - q[i] for i in range(4)], [p1[i] - q[i] for i in range(4)])*(dot_product(q, q)- mx**2)
       Num=Num1+Num2
       V[mu] = gx**3 * (Num/Denom)
   return V
#hulthen
def MatH(gx, mp, p1, p2, q):
   # Denominator terms
   denom1 = dot_product(p1, p1)-mp**2
   denom2 = mp**2- dot_product([p1[i] - q[i] for i in range(4)], [p1[i] - q[i] for i in range(4)])
   V = np.zeros(4)
   for mu in range(4):
       term1 = 2 * p1[mu] / denom1
       term3 = 2 * p2[mu] / denom2
       V[mu] = gx**3 * (term1 + term3)
   return V
#Total unpolorized square of matrix elemnt after doing polorization sum
#Columb 
def PertMatC(M):
   # Extract spatial components (j = 1,2,3)
   Mj = np.array(M[1:4])   # [Mx, My, Mz]
   # Compute spatial dot products
   Mj_dot_Mj = np.dot(Mj, Mj)          # Σ_j M^j M^j*
   # Apply the polarization sum formula
   return Mj_dot_Mj
#Hulthen
def PertMatH(M, q, mV):

    # Extract spatial components (j = 1,2,3)
    Mj = np.array(M[1:4])   # [Mx, My, Mz]
    qj = np.array(q[1:4])   # [qx, qy, qz]
    
    # Compute spatial dot products
    Mj_dot_Mj = np.dot(Mj, Mj)          # Σ_j M^j M^j*
    qj_dot_Mj = np.dot(qj, Mj)          # Σ_j q^j M^j
    
    # Compute |q|^2 = qx^2 + qy^2 + qz^2
    q_sq = np.dot(qj, qj)
    
    # Apply the polarization sum formula
    return Mj_dot_Mj - (qj_dot_Mj**2) / (q_sq + mV**2)

if __name__ == "__main__":
    mx_values = np.linspace(10, 1000, 1000)
    vel_values = [30, 300, 1000]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
   
    
    # Results storage
    coulomb_results = {}
    hulthen_results = {}
    
    for v_idx, v_rel_km_s in enumerate(vel_values):
        v_rel_km_s=30
        beta_rel = v_rel_km_s / 2.998e5
        V0 = g_x**2
        
        # Initialize storage for current velocity
        cs1c, cs2c, cs3c = [], [], []
        cs1h, cs2h, cs3h = [], [], []
        
        for mx in mx_values:
            mu = mx / 2
            S=4*mx**2
            Ylm = 1/np.sqrt(4*np.pi)
            a0=1/(mu*V0)#hbar=1 is set here
            #Mass of bound state
            #columb
            E1c=(mu * V0**2) / (2)
            MB1c=2*mx-E1c
            E2c=(mu * V0**2) / (2 * 2**2)
            MB2c=2*mx-E2c
            E3c=(mu * V0**2) / (2 * 3**2)
            MB3c=2*mx-E3c
           
            
            #energy and Momentum of emitted dark phton
            # we treat energy as a free parameter, take a numerical value which satisy the conditon written in paper and my notes
            #columb
            Wc1=10*mx
            Qc1=np.sqrt(Wc1**2)
            Wc2=10*mx
            Qc2=np.sqrt(Wc2**2)
            Wc3=10*mx
            Qc3=np.sqrt(Wc3**2)
            
            # COM energy
            s1c=(MB1c+Wc1)**2
            s2c=(MB2c+Wc2)**2
            s3c=(MB3c+Wc3)**2
            
            # Coulomb wavefunctions
            psi100c=2/(a0)**(3/2)
            psi200c=1/(2*a0**3)**(1/2)
            psi300c=2/(3*(3*a0**3)**(1/2))
            
        
            
            #Momentum vectors for calculating |M|^2
            #momentum of incoming 2 DM
            # In CM frame: p1 = (E, p), p2 = (E, -p) with |p| = (mx * beta) / 2
            E = mx
            p_mag = mx * beta_rel / 2
            p1 = [E, p_mag, 0, 0]   # Four-vector for particle 1
            p2 = [E, -p_mag, 0, 0]  # Four-vector for particle 2
            #momentu of photon in com
            #columb
            termc1=(4*mx**2+MB1c**2-mp**2)**2/(4*mx**2)-MB1c**2
            Q_cm_c1=np.sqrt(termc1)
            termc2=(4*mx**2+MB2c**2-mp**2)**2/(4*mx**2)-MB2c**2
            Q_cm_c2=np.sqrt(termc2)
            termc3=(4*mx**2+MB3c**2-mp**2)**2/(4*mx**2)-MB3c**2
            Q_cm_c3=np.sqrt(termc3)
          
            #momentum of emitted photon in rest fram eof bound state
            #Q=(E,0,0,Qz) is ususaly used basis for momentum, whcih obeys polorization rule e.q=0, where e is polorization vector
            #Columb
            q_vec1c = [Wc1, 0, 0, Qc1]
            q_vec2c = [Wc2, 0, 0, Qc2]
            q_vec3c = [Wc3, 0, 0, Qc3]
            
            #Extracting pertubative |M|^2
            #columbic
            V1c=MatC(g_x, mp, p1, p2, q_vec1c)
            V2c=MatC(g_x, mp, p1, p2, q_vec2c)
            V3c=MatC(g_x, mp, p1, p2, q_vec3c)
            MP1c=PertMatC(V1c)
            MP2c=PertMatC(V2c)
            MP3c=PertMatC(V3c)
          
            
            
            # Cross sections 
            #prefactors
            C1 = 1/(32*np.pi*MB1c*Wc1)*np.sqrt(s1c-S)/np.sqrt(s1c)
            C2 = 1/(32*np.pi*MB2c*Wc2)*np.sqrt(s2c-S)/np.sqrt(s2c) *2**5
            C3 = 1/(32*np.pi*MB3c*Wc3)*np.sqrt(s3c-S)/np.sqrt(s3c) *3**5
        
            
            # Coulomb
            M1 = np.sqrt(1/(2*mu))*psi100c
            M2 = np.sqrt(1/(2*mu))*psi200c
            M3 = np.sqrt(1/(2*mu))*psi300c
            cs1c.append(C1*M1**2*MP1c)
            cs2c.append(C2*M2**2*MP2c)
            cs3c.append(C3*M3**2*MP3c)
            
            
        
        # Store results
        coulomb_results[v_rel_km_s] = (cs1c, cs2c, cs3c)
       


# Plot Coulomb and Hulthen results with positioned and angled labels
plt.figure(figsize=(8, 8))
x = 2.5  # Line width
plt.plot(mx_values, cs1c, 'r-', label='n=1', linewidth=x)
plt.plot(mx_values, cs2c, 'g--',label='n=2', linewidth=x)
plt.plot(mx_values, cs3c, 'b-.',label='n=3', linewidth=x)

plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()

# Axis labels and formatting
plt.xlabel(r'$m_\chi$')
plt.ylabel(r'$\sigma_\mathrm{BD}$')
plt.yscale('log')
plt.xscale('log')
plt.grid(True, ls='--', alpha=0.7)
plt.tight_layout()
# auto-pick xvals for each line
lines = plt.gca().get_lines()


labelLines(lines, align=True,xvals=[500,400,500,500,500,500])
plt.text(0.07, 0.96, "(a)", transform=plt.gca().transAxes, fontsize=22)
plt.show()


