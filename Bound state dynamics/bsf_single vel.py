#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 12:09:23 2025

@author: goki
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import assoc_laguerre as al
from scipy.special import jacobi as jn
import matplotlib.pyplot as plt
from scipy.special import loggamma
from matplotlib.ticker import LogLocator, NullFormatter
from labellines import labelLines

# Constants


lam_x = 0.8
g_x = 0.7   # Coupling constant
# global plot text size 
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


#Four vector dot product
def dot_product(a, b):
    """Minkowski dot product for four-vectors (+,-,-,-)"""
    return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
# Perturbative amplitudes
#matrix element in four vector form
#columb
def MatC(gx, p1, p2, q):
   # Denominator terms
   denom1 = dot_product(p1, p1)
   denom2 = dot_product([p1[i] + p2[i] for i in range(4)], [p1[i] + p2[i] for i in range(4)]) 
   denom3 =  dot_product([p1[i] - q[i] for i in range(4)], [p1[i] - q[i] for i in range(4)])
   V = np.zeros(4)
   for mu in range(4):
       term1 = 2 * p1[mu] / denom1
       term2 = 2 * (p1[mu] - p2[mu]) / denom2
       term3 = 2 * p2[mu] / denom3
       V[mu] = gx**3 * (term1 + term2 - term3)
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

if __name__ == "__main__":
    mxV = np.linspace(10, 1000, 100)
    vel_values = [30, 300, 1000]
    
    
   
    
    # Initialize storage lists
    cs1c, cs2c, cs3c = [], [], []
    cs1h, cs2h, cs3h = [], [], []
    mx_value=[]
    

       
        
    for mx in mxV:
            mx_value.append(mx)
            v_rel_km_s=3000
            beta_rel = v_rel_km_s / 2.998e5
            V0 = g_x**2
            mu = mx / 2
            a0=1/(mu*V0)#hbar=1 is set here
            s=4*mx**2
            Ek=1/2*mu*v_rel_km_s**2
            Ylm = 1/np.sqrt(4*np.pi)
            # Energy and Mass of bound state
            #columb
            E1c=(mu * V0**2) / (2)
            MB1c=2*mx-E1c
            E2c=(mu * V0**2) / (2 * 2**2)
            MB2c=2*mx-E2c
            E3c=(mu * V0**2) / (2 * 3**2)
            MB3c=2*mx-E3c
            
            #energy and Momentum of emitted dark phton
            #columb
            Wc1=(s-MB1c**2)/(2*MB1c)
            Qc1=np.sqrt(Wc1**2)
            Wc2=(s-MB2c**2)/(2*MB2c)
            Qc2=np.sqrt(Wc2**2)
            Wc3=(s-MB3c**2)/(2*MB3c)
            Qc3=np.sqrt(Wc3**2)
           
            #Wavefunction at r=0
            psi100c=2/(a0)**(3/2)
            psi200c=1/(2*a0**3)**(1/2)
            psi300c=2/(3*(3*a0**3)**(1/2))
            #Scattering states(at r= 0)
            k=mu*beta_rel
            #Columb
            D=V0/beta_rel
            log_term = (D * np.pi / 2) + loggamma(1 - 1j * D) # we caluclate using log and take exp, this way, we have numerical stablity
            PsiScatC=np.exp(log_term) # after applying exp to log term we recover original eqn np.exp((D*np.pi)/2)*g(1-1j*D)
            
            #Momentum vectors for calculating |M|^2
            #momentum of incoming 2 DM
            # In CM frame: p1 = (E, p), p2 = (E, -p) with |p| = (mx * beta) / 2
            E = mx
            p_mag = mx * beta_rel / 2
            p1 = [E, p_mag, 0, 0]   # Four-vector for particle 1
            p2 = [E, -p_mag, 0, 0]  # Four-vector for particle 2
           
            #momentum of emitted photon in rest fram eof bound state
            #Q=(E,0,0,Qz) is ususaly used basis for momentum, whcih obeys polorization rule e.q=0, where e is polorization vector
            #Columb
            q_vec1c = [Wc1, 0, 0, Qc1]
            q_vec2c = [Wc2, 0, 0, Qc2]
            q_vec3c = [Wc3, 0, 0, Qc3]
           
            #Extracting pertubative |M|^2
            #columbic
            V1c=MatC(g_x, p1, p2, q_vec1c)
            V2c=MatC(g_x, p1, p2, q_vec2c)
            V3c=MatC(g_x,  p1, p2, q_vec3c)
            MP1c=PertMatC(V1c)
            MP2c=PertMatC(V2c)
            MP3c=PertMatC(V3c)
            #Hulthen
            
            
            
            # Cross sections
            #Energy diff
            De1c=(Ek-E1c)
            De2c=(Ek-E2c)
            De3c=(Ek-E3c)
            
           
            #prefactors
            C1 = 1/(16*np.pi*mx**2)*(De1c)/(mu*v_rel_km_s**3)
            C2 = 1/(16*np.pi*mx**2)*(De2c)/(mu*v_rel_km_s**3)
            C3 = 1/(16*np.pi*mx**2)*(De3c)/(mu*v_rel_km_s**3)
            #hultheh prefactor
           
            
            # Coulomb
            M1 = (1/np.sqrt(2*mu))*psi100c*PsiScatC
            M2 = (1/np.sqrt(2*mu))*psi200c*PsiScatC
            M3 = (1/np.sqrt(2*mu))*psi300c*PsiScatC
            cs1c.append(C1*np.abs(M1)**2*MP1c)
            cs2c.append(C2*np.abs(M2)**2*MP2c)
            cs3c.append(C3*np.abs(M3)**2*MP3c)
            
           
        
        # Plot for Coulomb BSF (current velocity)
x = 2.5

# Coulomb BSF
plt.figure(figsize=(8, 8))
plt.plot(mx_value, cs1c, 'r-', label='n=1', linewidth=x)
plt.plot(mx_value, cs2c, 'g--', label='n=2', linewidth=x)
plt.plot(mx_value, cs3c, 'b', label='n=3', linewidth=x)
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.xlabel(r'$m_\chi$')
plt.ylabel(r'$\sigma_\mathrm{BSF}$')
# Clean grid (major only, minor lighter)
plt.grid(which="major", linestyle="--", linewidth=0.8, alpha=0.7)
plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
plt.yscale('log')
plt.xscale('log')
# Label directly along the lines
labelLines(plt.gca().get_lines(), align=True,xvals=[100, 100, 100], fontsize=16)
plt.text(0.07, 0.96, "(a)", transform=plt.gca().transAxes, fontsize=22)
plt.show()
