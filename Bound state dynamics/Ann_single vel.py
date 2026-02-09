#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 13:15:11 2025

@author: goki
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import assoc_laguerre as al
from scipy.special import jacobi as jn
from scipy.special import loggamma
import matplotlib.pyplot as plt
from labellines import labelLines

mh = 125000  # MeV (mass of Higgs boson)
mp = 0.05       # MeV (mass of mediator)
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

def PetMatAnn(mf):# feed in the mass of fermion DM anhillates into and lam give coupling constnat i.e lam xh for DM anhillation into fermion or lam hp for mediator anhillation into fermion
    s=4*mx**2
   
    lam_hf=mf
    M1=2*(s-4*mf**2)
    M2=((lam_xh*lam_hf)**2/(s-mh**2))**2
    return M1*M2 # see simpler realsiastio of DM paper
def AnnDecCS(mf,MatSq):
    Prefact2=1/(64*mx**2*np.pi)
    CS=Prefact2*np.sqrt(4*mx**2-4*mf**2)*MatSq
    return CS
def pertAnnPion(mp):
    s=4*mx**2
    lam_hp=mp**2
    pre=lam_xh*lam_hp/2
    M_sq=(pre/(s-mh**2))**2
    return M_sq

def AnnDecCS_p(mp,MatSq):# M_SM is mass of SM particle DM is anhillated into , electron , muon or pion
    Prefact2=1/(64*mx**2*np.pi)
    
    CS=Prefact2*np.sqrt(4*mx**2-4*mp**2)*MatSq
    return CS


if __name__ == "__main__":
    mx_values = np.linspace(10, 1000, 1000)
    vel_values = [30, 300, 1000]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(8, 8))
   
    
    # Results storage
    coulomb_results = {}
    hulthen_results = {}
    
    
        
        
        # Initialize storage for current velocity
        # --- initialize empty lists for annihilation cross sections ---
    CS_Ann_C_X_el = []
    CS_Ann_C_Pc = []
    CS_Ann_C_X_Mu = []
    CS_Ann_C_Pn= []

    CS_Ann_H_X_el = []
    CS_Ann_H_Pc = []
    CS_Ann_H_X_Mu = []
    CS_Ann_H_Pn = []
        
    for mx in mx_values:
            mu = mx / 2
            Ylm = 1/np.sqrt(4*np.pi)
            v_rel_km_s=30
            beta_rel = v_rel_km_s / 2.998e5
            p=mx/2*v_rel_km_s
            V0 = g_x**2
            #Scattering states(at r= 0)
            k=mu*beta_rel
            #Columb
            D=V0/beta_rel
            log_term = (D * np.pi / 2) + loggamma(1 - 1j * D) # we caluclate using log and take exp, this way, we have numerical stablity
            PsiScatC=np.exp(log_term) # after applying exp to log term we recover original eqn np.exp((D*np.pi)/2)*g(1-1j*D)
            #hulthen
            Lamb=1
            ep1=(2*k*V0*mp)/(beta_rel*mp**2)
            ep2=k**2/mp**2
            epi=np.sqrt(ep1-ep2)
            a= Lamb-(1j*k)/mp+epi
            b= Lamb-(1j*k)/mp-epi
            c=2*Lamb
            PsiScath = np.exp(loggamma(c - a) + loggamma(c - b) - loggamma(c - a - b)) # use log gamma for numerical stablity
            # Anhillation and decay calculation
           # Matrix elements
           
            M_ele=0.511
            M_mu=105.658
            M_Pc=139.57 # mass of charged pions
            M_p0=135 # mass of neutral pion
            #Anhillation
            #these are square of matrix elements
            #columb
            # here we multipy extra np.abs(PsiScatC) ** 2 whci is actually mod square(psi_scat at r=0) i.e sommerfiedl enhancement factor
            Mat_el_Xc=PetMatAnn(M_ele)*np.abs(PsiScatC) ** 2 # matrix element sq for electron anhillation =  (psi_k*M_pert)**2
            Mat_mu_Xc=PetMatAnn(M_mu)*np.abs(PsiScatC) ** 2 # matrix element sq for muon anhillation =  (psi_k*M_pert)**2
            Mat_Pc_Xc=pertAnnPion(M_Pc)*np.abs(PsiScatC) ** 2 # matrix element sq for charged pion anhillation =  (psi_k*M_pert)**2
            
            
            
           
           
            # Coulomb
            CS_Ann_C_X_el.append(AnnDecCS(M_ele, Mat_el_Xc))
            CS_Ann_C_X_Mu.append(AnnDecCS(M_mu, Mat_mu_Xc))
            CS_Ann_C_Pc.append(AnnDecCS_p(M_Pc, Mat_Pc_Xc))
           
        
        
        
       
    
  
         
x = 2.5  # line width

# Coulomb BSF
plt.figure(figsize=(8, 8))
plt.plot(mx_values, CS_Ann_C_X_el, 'r-', label= "$e^{{\pm}}$", linewidth=x)
plt.plot(mx_values, CS_Ann_C_X_Mu, 'g--', label="$\mu^{{\pm}}$", linewidth=x)
plt.plot(mx_values, CS_Ann_C_Pc, 'b', label="$\pi^{{\pm}}$", linewidth=x)
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.xlabel(r'$m_\chi$')
plt.ylabel(r'$\sigma_\mathrm{Ann}$')
# Clean grid (major only, minor lighter)
plt.grid(which="major", linestyle="--", linewidth=0.8, alpha=0.7)
plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
plt.yscale('log')
plt.xscale('log')
labelLines(plt.gca().get_lines(), align=True,xvals=[800, 800, 800], fontsize=16)
plt.text(0.07, 0.96, "(a)", transform=plt.gca().transAxes, fontsize=22)
plt.show()



