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

# Constants
mx=150
mh = 125000  # MeV (mass of Higgs boson)
mp = 0.05       # MeV (mass of mediator)
lam_xh = 1e-3
lam_hp = 0.3
lam_x = 0.8
g_x = 0.7 

conv = 0.389379e-21
# global plot text size 
plt.rcParams.update({
    'font.size': 18,          # Base font size
    'axes.titlesize': 18,     # Subplot titles
    'axes.labelsize': 21,     # Axis labels
    'xtick.labelsize': 18,    # X-axis ticks
    'ytick.labelsize': 18,    # Y-axis ticks
    'legend.fontsize': 18,    # Legend text
    'figure.titlesize': 18    # Main title
})




if __name__ == "__main__":
    mx_values = np.linspace(200, 1000, 1000)
    vel_values = [30, 300, 1000]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
   
    
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
        
def CS(v_rel_km_s):
            
            mu = mx / 2
            Ylm = 1/np.sqrt(4*np.pi)
           
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
            def PetMatAnn(mf):# feed in the mass of fermion DM anhillates into and lam give coupling constnat i.e lam xh for DM anhillation into fermion or lam hp for mediator anhillation into fermion
                s=4*mx**2
               
                lam_hf=mf
                M1=2*(s-4*mf**2)
                M2=((lam_xh*lam_hf)**2/(s-mh**2))**2
                return M1*M2 # see simpler realsiastio of DM paper
            def AnnDecCS(mf,MatSq):
                Prefact2=1/(64*mx**2*np.pi)
                CS=Prefact2*np.sqrt(4*mx**2-4*mf**2)*MatSq
                return CS*conv
            def pertAnnPion(mp):
                s=4*mx**2
                lam_hp=mp**2
                pre=lam_xh*lam_hp/2
                M_sq=(pre/(s-mh**2))**2
                return M_sq

            def AnnDecCS_p(mp,MatSq):# M_SM is mass of SM particle DM is anhillated into , electron , muon or pion
                Prefact2=1/(64*mx**2*np.pi)
                
                CS=Prefact2*np.sqrt(4*mx**2-4*mp**2)*MatSq
                return CS*conv
           
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
            CS_Ann_C_X_el=AnnDecCS(M_ele, Mat_el_Xc)
            CS_Ann_C_X_Mu=AnnDecCS(M_mu, Mat_mu_Xc)
            CS_Ann_C_Pc=AnnDecCS_p(M_Pc, Mat_Pc_Xc)
           
            return CS_Ann_C_Pc
def intergrand(v_rel_km_s):
    cs = CS(v_rel_km_s)
    x = 3e6  # For galaxies, see https://arxiv.org/pdf/1006.2518 indirect search section
    prefact = x ** (3 / 2) / (2 * np.sqrt(np.pi))
    return prefact * v_rel_km_s ** 2 * np.exp(-1 / 4 * x * v_rel_km_s ** 2) * cs
ThermCS,_=quad(intergrand, 0, np.inf)
print(ThermCS)
                
        
        
        
       
    
  
   