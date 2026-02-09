#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:34:11 2025

@author: goki
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 12:33:24 2025

@author: goki
"""



import numpy as np
from scipy.integrate import quad
from scipy.special import assoc_laguerre as al
from scipy.special import jacobi as jn
from scipy.special import loggamma
import matplotlib.pyplot as plt

# Constants
mx=150 # MeV dark matter mass
mh = 125000  # MeV (mass of Higgs boson)
mp = 0.05       # MeV (mass of mediator)
lam_xh = 0.0001
lam_hp = 0.3
lam_x = 0.8
g_x = 0.7
conv = 0.389379e-21
V0=g_x**2
M_ele=0.511
M_mu=105.658
M_Pc=139.57 # mass of charged pions
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




def CS_Calc(v_rel_cm_s):
    #columb
    #Wavefunction at r=0
    mx=350
    mu=mx/2
    a0=1/(mu*V0)#hbar=1 is set here                         
 
    psi1c = 2/(a0**(3/2))
    psi2c = psi1c / (2**1.5)   # psi1c / n^{3/2}
    psi3c = psi1c / (3**1.5)

    def PetMatAnn(mf):# feed in the mass of fermion DM anhillates into and lam give coupling constnat i.e lam xh for DM anhillation into fermion or lam hp for mediator anhillation into fermion
        s=4*mx**2
       
        lam_hf=mf
        M1=2*(s-4*mf**2)
        M2=((lam_xh*lam_hf)**2/(s-mh**2))**2
        return M1*M2 # see simpler realsiastio of DM paper
    def AnnDecCS(mf,MatSq):
        Prefact2=1/(64*mx**2*np.pi)
        CS=Prefact2*np.sqrt(mx**2-4*mf**2)*MatSq
        return CS*conv
    def pertAnnPion(mp):
        s=4*mx**2
        lam_hp=mp**2
        pre=lam_xh*lam_hp/2
        M_sq=(pre/(s-mh**2))**2
        return M_sq

    def AnnDecCS_p(mp,MatSq):# M_SM is mass of SM particle DM is anhillated into , electron , muon or pion
        Prefact2=1/(64*mx**2*np.pi)
        
        CS=Prefact2*np.sqrt(mx**2-4*mp**2)*MatSq
        return CS*conv
    
    Mat_el_Xc1=PetMatAnn(M_ele)*np.abs(psi1c) ** 2 # matrix element sq for electron anhillation =  (psi_k*M_pert)**2
    CS_ce1=AnnDecCS(M_ele, Mat_el_Xc1)
    
    Mat_el_Xc2=PetMatAnn(M_ele)*np.abs(psi2c) ** 2 # matrix element sq for electron anhillation =  (psi_k*M_pert)**2
    CS_ce2=AnnDecCS(M_ele, Mat_el_Xc2)
   
    Mat_el_Xc3=PetMatAnn(M_ele)*np.abs(psi3c) ** 2 # matrix element sq for electron anhillation =  (psi_k*M_pert)**2
    CS_ce3=AnnDecCS(M_ele, Mat_el_Xc3)
   
    Mat_mu_Xc1=PetMatAnn(M_mu)*np.abs(psi1c) ** 2 # matrix element sq for muon anhillation =  (psi_k*M_pert)**2
    CS_cm1=AnnDecCS(M_mu, Mat_mu_Xc1)
    
    Mat_mu_Xc2=PetMatAnn(M_mu)*np.abs(psi2c) ** 2 # matrix element sq for muon anhillation =  (psi_k*M_pert)**2
    CS_cm2=AnnDecCS(M_mu, Mat_mu_Xc2)
    
    Mat_mu_Xc3=PetMatAnn(M_mu)*np.abs(psi3c) ** 2 # matrix element sq for muon anhillation =  (psi_k*M_pert)**2
    CS_cm3=AnnDecCS(M_mu, Mat_mu_Xc3)
   
    Mat_Pc_Xc1=pertAnnPion(M_Pc)*np.abs(psi1c) ** 2 # matrix element sq for charged pion anhillation =  (psi_k*M_pert)**2
    CS_cp1=(AnnDecCS_p(M_Pc, Mat_Pc_Xc1))
   
    Mat_Pc_Xc2=pertAnnPion(M_Pc)*np.abs(psi2c) ** 2 # matrix element sq for charged pion anhillation =  (psi_k*M_pert)**2
    CS_cp2=(AnnDecCS_p(M_Pc, Mat_Pc_Xc2))
   
    Mat_Pc_Xc3=pertAnnPion(M_Pc)*np.abs(psi3c) ** 2 # matrix element sq for charged pion anhillation =  (psi_k*M_pert)**2
    CS_cp3=(AnnDecCS_p(M_Pc, Mat_Pc_Xc3))


    return CS_cp2

def intergrand(v_rel_km_s):
    cs = CS_Calc(v_rel_km_s)
    x = 3e6  # For galaxies, see https://arxiv.org/pdf/1006.2518 indirect search section
    prefact = x ** (3 / 2) / (2 * np.sqrt(np.pi))
    return prefact * v_rel_km_s ** 2 * np.exp(-1 / 4 * x * v_rel_km_s ** 2) * cs
ThermCS,_=quad(intergrand, 0, np.inf)
print(ThermCS)
cs = CS_Calc(3000)
print(cs)

                
                
                
                
               
    