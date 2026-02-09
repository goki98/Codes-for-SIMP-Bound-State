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
from labellines import labelLines


lam_x = 0.8
g_x = 0.7 # Coupling constant
# global plot text size 
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
    mx_values = np.linspace(10, 1000, 1000)
    vel_values = [30, 300, 1000]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
   
    
    # Results storage
    coulomb_results = {}
    hulthen_results = {}
    
   
        
        # Initialize storage for current velocity
    cs1c, cs2c, cs3c = [], [], []
    cs1h, cs2h, cs3h = [], [], []
    mx_value=[]
        
    for mx in mx_values:
            mx_value.append(mx)
            v_rel_km_s=30
            beta_rel = v_rel_km_s / 2.998e5
            V0 = g_x**2
            mu = mx / 2
            s=4*mx**2
            Ylm = 1/np.sqrt(4*np.pi)
            #Mass of bound state
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
           
            # Coulomb wavefunctions
            a0=1/(mu*V0)#hbar=1 is set here
            psi100c=2/(a0)**(3/2)*Ylm
            psi200c=1/(2*a0**3)**(1/2)*Ylm
            psi300c=2/(3*(3*a0**3)**(1/2))*Ylm
            
            
           
            
            #Momentum vectors for calculating |M|^2
            #momentum of incoming 2 DM
            # In CM frame: p1 = (E, p), p2 = (E, -p) with |p| = (mx * beta) / 2
            E = mx
            p_mag = mx * beta_rel / 2
            p1 = [E, p_mag, 0, 0]   # Four-vector for particle 1
            p2 = [E, -p_mag, 0, 0]  # Four-vector for particle 2
            #momentu of photon in com
            #columb
            termc1=(4*mx**2+MB1c**2)**2/(4*mx**2)-MB1c**2
            Q_cm_c1=np.sqrt(termc1)
            termc2=(4*mx**2+MB2c**2)**2/(4*mx**2)-MB2c**2
            Q_cm_c2=np.sqrt(termc2)
            termc3=(4*mx**2+MB3c**2)**2/(4*mx**2)-MB3c**2
            Q_cm_c3=np.sqrt(termc3)
           
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
            V3c=MatC(g_x, p1, p2, q_vec3c)
            MP1c=PertMatC(V1c)
            MP2c=PertMatC(V2c)
            MP3c=PertMatC(V3c)
           
            
            
            # Cross sections 
            # Q_cm_n , where n is the n th state of bound state in final
            #prefactors
            C21 = 1/(16*np.pi*mx**2)*(Q_cm_c1/(2*mx))
            C32 = 1/(16*np.pi*mx**2)*(Q_cm_c2/(2*mx))
            C31 = 1/(16*np.pi*mx**2)*(Q_cm_c1/(2*mx))
            
            
            # Coulomb
            M21c = np.sqrt(2/mu)*psi100c*psi200c
            M32c = np.sqrt(2/mu)*psi200c*psi300c
            M31c = np.sqrt(2/mu)*psi300c*psi100c
            cs1c.append(C21*M21c**2*MP1c)
            cs2c.append(C32*M32c**2*MP2c)
            cs3c.append(C31*M31c**2*MP3c)
            
           
       # Plot for Coulomb BSF (current velocity)
    x=2.5
    plt.figure(figsize=(8, 8))
    plt.plot(mx_value, cs1c, 'r-', label=r"$2 \rightarrow 1$",linewidth=x)
    plt.plot(mx_value, cs2c, 'g--', label=r"$3 \rightarrow 2$",linewidth=x)
    plt.plot(mx_value, cs3c, 'b', label=r"$3 \rightarrow 1$",linewidth=x)
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    plt.minorticks_on()
    plt.xlabel(r'$m_\chi$ ')
    plt.ylabel(r'$\Gamma_\mathrm{Trans}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which='major', ls='--', alpha=0.7)  # Only major grid (log scale)
    labelLines(plt.gca().get_lines(), align=True,xvals=[100, 100, 100], fontsize=16)
 
    plt.tight_layout()
    plt.show()

       
