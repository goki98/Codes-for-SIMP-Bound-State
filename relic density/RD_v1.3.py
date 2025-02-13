#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:48:32 2024

@author: goki
"""

import numpy as np
from hepunits import GeV as m
from hepunits import MeV as m2
from scipy.integrate import quad
import matplotlib.pyplot as plt
# parameters
rd=[]
mdm=[]
xf1=[]
xf2=[]
xf3=[]
for M_c in np.arange(0 * m2, 2000 * m2, 1 * m2):
    
    """ M_c_g=M_c*1.8*10**(-24) # GeV in unit of grams
    L_ch = 0.01
    L_c = 0.1
    v_h = 246*m
    L_cp = 0.0001
    v_p = 60*M_c
    mu_cp = 0.1*M_c
    c = 0.1
    s = 0.99
    m_1 = 25*M_c
    mu_c = 0.005*M_c
    m_2 = 125*m
    # matrix element
    M=(((L_ch * v_h * c - (L_cp * v_p + mu_cp) * s))*(L_ch * v_h * s + (L_cp * v_p + mu_cp) * c))/(4*M_c**2-m_1**2)
    # thermal avg cross section
    M_net=M**2
    cs_th = (np.sqrt(5) / (192 * np.pi * M_c**3)) * M_net"""

    #freeze out temp
    gs=10.75
    MPl=1.2*10**22
    g_dm=1
    g_dm_s=g_dm**2/np.sqrt(gs)
    c1=1 #c(c+1)**2 term in eqn
    c2=5
    c3=10
    cs_t=2*10**6*(m)**-5
    X_1=0.0024*g_dm_s*c1*MPl*M_c**4*cs_t
    X_f1=1/2*np.log(X_1)-2*np.log(1/2*np.log(X_1))
    X_2=0.0024*g_dm_s*c2*MPl*M_c**4*cs_t
    X_f2=1/2*np.log(X_2)-2*np.log(1/2*np.log(X_2))
    X_3=0.0024*g_dm_s*c3*MPl*M_c**4*cs_t
    X_f3=1/2*np.log(X_3)-2*np.log(1/2*np.log(X_3))
    xf1.append(X_f1)
    xf2.append(X_f2)
    xf3.append(X_f3)
    
    #relic density
    
    Omega=(0.33/gs**(3/4))*((1*m2*10**3)/M_c)*X_f1**2*np.sqrt(((1)/cs_t))
    rd.append(Omega)
    mdm.append(M_c)
    #print(M_c,X_f,Omega)
plt.scatter(mdm, xf1, s=5 ,c="blue", alpha=1, label="c=1")
plt.scatter(mdm, xf2, s=5 ,c="red", alpha=1, label="c=5")
plt.scatter(mdm, xf3, s=5 ,c="green", alpha=1, label="c=10")

plt.xlabel("M_c [MeV]")

plt.ylabel("Xf=Tf/Mdm")

plt.title("Mdm vs Xf with c=1,5,10")

plt.legend()

plt.show()