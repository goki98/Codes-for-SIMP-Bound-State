#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 15:30:58 2025

@author: goki
"""

import numpy as np
from scipy.integrate import quad
from labellines import labelLines
import matplotlib.pyplot as plt
from scipy.special import loggamma
from scipy.integrate import dblquad
lam_x = 0.8
g_x = 0.7
mx=30
epi=1j*1e-20# +i epsilon factor
#Four vector dot product
def dot_product(a, b):
    """Minkowski dot product for four-vectors (+,-,-,-)"""
    return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
#for process 7
def PertMatCR(M):
   # Extract spatial components (j = 1,2,3)
   Mj = np.array(M[1:4])   # [Mx, My, Mz]
   # Compute spatial dot products
   Mj_dot_Mj = np.dot(Mj, Mj)          # Σ_j M^j M^j*
   # Apply the polarization sum formula
   return Mj_dot_Mj

def pertmat(p1,p2,p3,p4,k1,k2):
    #M1
    P1 = [p1[i] + p2[i] + p3[i] for i in range(4)]

    M1=lam_x**2/(dot_product(P1, P1)-mx**2)
    #M2
    P2=[p1[i]+p2[i]-k1[i] for i in range(4)]
    M2=lam_x**2/(dot_product(P2, P2)-mx**2)
    #M3
    P3=[k1[i]+k2[i]-p1[i] for i in range(4)] #q in diagram form notes
    t31=[p2[i]+P3[i] for i in range(4)]
    t32=[-p3[i]-p4[i] for i in range(4)]
    d31=[p2[i]-P3[i] for i in range(4)]
    d32=[P3[i]+p1[i] for i in range(4)]
    M3=(g_x**2 *dot_product(t31, t32)*lam_x)/((dot_product(d31, d31))*(dot_product(d32, d32)-mx**2+epi))
    #m4
    M4=(lam_x*g_x**2 *dot_product([p3[i]-p4[i]for i in range(4)],[p2[i]-k1[i]-k2[i]for i in range(4)]))/(dot_product([p2[i]-k1[i]for i in range(4)], [p2[i]-k1[i]for i in range(4)])*dot_product([p3[i]+p4[i]for i in range(4)], [p3[i]+p4[i]for i in range(4)]))
    #M5
    k5=[p4[i]-k2[i]for i in range(4)]
    q=[p3[i]-k5[i]for i in range(4)]
    num5=(lam_x*g_x**2*dot_product(([p3[i]+q[i]for i in range(4)]),([-p4[i]-k2[i]for i in range(4)])))
    den5=((dot_product(([p2[i]-k1[i]for i in range(4)]),([p2[i]-k1[i]for i in range(4)]))-mx**2+epi)*dot_product(([p3[i]-q[i]for i in range(4)]), ([p3[i]-q[i]for i in range(4)])))
    M5=num5/den5
    #M6
    P6=[p3[i]+p2[i]+p4[i]for i in range(4)]
    q6=[P6[i]+p1[i]for i in range(4)]
    num6=lam_x*g_x**3*dot_product(([P6[i]+p1[i]for i in range(4)]), ([k1[i]-k2[i]for i in range(4)]))
    den6=(dot_product(([p1[i]-q6[i]for i in range(4)]), ([p1[i]-q6[i]for i in range(4)]))-mx**2+epi)*(dot_product(([p1[i]+P6[i]for i in range(4)]), ([p1[i]+P6[i]for i in range(4)])))
    M6=num6/den6 
    #M7
    P7=[p3[i]+p4[i]-k2[i]for i in range(4)]
    q7=[p2[i]-P7[i]for i in range(4)]
    num7=g_x**2*dot_product(([p1[i]+k1[i]for i in range(4)]),([P7[i]-p2[i]for i in range(4)]))*lam_x
    den7=dot_product(([p1[i]-k1[i]for i in range(4)]),([p1[i]-k1[i]for i in range(4)]))*(dot_product(([p2[i]-q7[i]for i in range(4)]),([p2[i]-q7[i]for i in range(4)])))
    M7=num7/den7
    
    #M8
    k8=[p3[i]-p4[i]for i in range(4)]
    q8=[p2[i]-k8[i]for i in range(4)]
    P8=[p1[i]+q8[i]for i in range(4)]
    num8=g_x**4*dot_product(([p4[i]-p3[i]for i in range(4)]),([p2[i]-q8[i]for i in range(4)]))*dot_product(([p1[i]-q8[i]for i in range(4)]),([k1[i]-k2[i]for i in range(4)]))
    den8=dot_product(([p3[i]+p4[i]for i in range(4)]),([p3[i]+p4[i]for i in range(4)]))*(dot_product(([p1[i]-P8[i]for i in range(4)]),([p1[i]-P8[i]for i in range(4)]))-mx**2+epi)*dot_product(([p1[i]+q8[i]for i in range(4)]),([p1[i]+q8[i]for i in range(4)]))
    M8=num8/den8
    
    #M9
    p9=[p3[i]+p4[i]for i in range(4)]
    k9=[p9[i]-k2[i]for i in range(4)]
    q9=[p2[i]-k9[i]for i in range(4)]
    num9=g_x**6*dot_product(([p1[i]+k1[i]for i in range(4)]),([k9[i]-p2[i]for i in range(4)]))*dot_product(([q9[i]-p2[i]for i in range(4)]),([p9[i]-k2[i]for i in range(4)]))*dot_product(([p3[i]-p4[i]for i in range(4)]),([k9[i]-k2[i]for i in range(4)]))
    den9=dot_product(([p1[i]-k1[i]for i in range(4)]),([p1[i]-k1[i]for i in range(4)]))*(dot_product(([p2[i]-q9[i]for i in range(4)]),([p2[i]-q9[i]for i in range(4)]))-mx**2+epi)*dot_product(([p3[i]+p4[i]for i in range(4)]),([p3[i]+p4[i]for i in range(4)]))
    M9=num9/den9
    
    #m10
  
    k10=[p4[i]-k2[i]for i in range(4)]
    q10=[p3[i]-k10[i]for i in range(4)]
    p10=[p2[i]-q10[i]for i in range(4)]
    num10=g_x**6*dot_product(([p1[i]+k1[i]for i in range(4)]),([q10[i]-p10[i]for i in range(4)]))*dot_product(([p10[i]-p2[i]for i in range(4)]),([p3[i]+k10[i]for i in range(4)]))*dot_product(([p3[i]+q10[i]for i in range(4)]),([p4[i]-k2[i]for i in range(4)]))
    den10=dot_product(([p1[i]-k1[i]for i in range(4)]),([p1[i]-k1[i]for i in range(4)]))*(dot_product(([p2[i]-p10[i]for i in range(4)]),([p2[i]-p10[i]for i in range(4)]))-mx**2+epi)*dot_product(([p3[i]-q10[i]for i in range(4)]),([p3[i]-q10[i]for i in range(4)]))
    M10=num10/den10
    
    #m11
  
    p11=[p2[i]+p3[i]+p4[i]for i in range(4)]
    q11=[p11[i]+p1[i]for i in range(4)]
    
    num11=g_x**2*lam_x*dot_product(([p1[i]+q11[i]for i in range(4)]),([k1[i]-k2[i]for i in range(4)]))
    den11=dot_product(([p1[i]+q11[i]for i in range(4)]),([p1[i]-q11[i]for i in range(4)]))*(dot_product(([p1[i]-q11[i]for i in range(4)]),([p1[i]-q11[i]for i in range(4)]))-mx**2+epi)
    M11=num11/den11
    
    #M12
    P12=[p1[i]-k1[i]for i in range(4)]
    Q12=[p2[i]+p3[i]+p4[i]for i in range(4)]
    num12=g_x**2*lam_x*dot_product(([p1[i]+k1[i]for i in range(4)]), ([Q12[i]-k2[i]for i in range(4)]))
    den12 = dot_product(P12, P12) * (dot_product([P12[i]+k2[i] for i in range(4)], [P12[i]+k2[i] for i in range(4)]) - mx**2 + epi)

    M12=num12/den12
    
    #M13
    Q13=[p3[i]+p4[i]-k2[i]for i in range(4)]
    P13=[p2[i]-Q13[i]for i in range(4)]
    num13=g_x**2*lam_x*dot_product(([p1[i]+k1[i]for i in range(4)]), ([Q13[i]-k2[i]for i in range(4)]))
    den13=dot_product(([p1[i]-k1[i]for i in range(4)]), ([p1[i]-k1[i]for i in range(4)]))*(dot_product(([P13[i]-p2[i]for i in range(4)]), ([P13[i]-p2[i]for i in range(4)]))-mx**2+epi)
    M13=num13/den13
    
    #M14
    k14=[p3[i]-p4[i]for i in range(4)]
    p14=[p2[i]-k14[i]for i in range(4)]
    q14=[p1[i]+p14[i]for i in range(4)]
    num14=g_x**4*dot_product(([p1[i]+p14[i]for i in range(4)]), ([k1[i]-k2[i]for i in range(4)]))*dot_product(([p2[i]-p14[i]for i in range(4)]), ([p4[i]-p3[i]for i in range(4)]))
    den14=dot_product(([p1[i]+p14[i]for i in range(4)]), ([p1[i]+p14[i]for i in range(4)]))*(dot_product(([p1[i]-q14[i]for i in range(4)]), ([p1[i]-q14[i]for i in range(4)]))-mx**2+epi)*dot_product(([p2[i]+p14[i]for i in range(4)]), ([p2[i]+p14[i]for i in range(4)]))
    M14=num14/den14
    
    #M15
    p15=[p3[i]+p4[i]for i in range(4)]
    k15=[p15[i]-k2[i]for i in range(4)]
    q15=[p2[i]-k15[i]for i in range(4)]
    num15=g_x**4*dot_product([p1[i]+k1[i]for i in range(4)], [k15[i]-p2[i]for i in range(4)])*dot_product([p3[i]-p4[i]for i in range(4)], [k15[i]-k2[i]for i in range(4)])
    den15=dot_product(([p1[i]-k1[i]for i in range(4)]), ([p1[i]-k1[i]for i in range(4)]))*(dot_product(([p2[i]-q15[i]for i in range(4)]), ([p2[i]-q15[i]for i in range(4)]))-mx**2+epi)*dot_product(([p3[i]+p4[i]for i in range(4)]), ([p3[i]+p4[i]for i in range(4)]))
    M15=num15/den15
    
    #M16
    q16=[p4[i]-k2[i]for i in range(4)]
    k16=[p3[i]-q16[i]for i in range(4)]
    p16=[p2[i]-k16[i]for i in range(4)]
    num16=g_x**3*dot_product([k16[i]-p2[i]for i in range(4)], [p1[i]+k1[i]for i in range(4)])*dot_product([k16[i]-p3[i]for i in range(4)], [k2[i]-p4[i]for i in range(4)])
    den16=dot_product(([p1[i]-k1[i]for i in range(4)]), ([p1[i]-k1[i]for i in range(4)]))*(dot_product(([p2[i]-p16[i]for i in range(4)]), ([p2[i]-p16[i]for i in range(4)]))-mx**2+epi)*dot_product(([k16[i]-p3[i]for i in range(4)]), ([k16[i]-p3[i]for i in range(4)]))
    M16=num16/den16
    
    #m17
    p17=[p3[i]-p4[i]for i in range(4)]
    q17=[p2[i]+p1[i]-p17[i]for i in range(4)]
    num17 = g_x**4 * dot_product([p3[i] - p4[i] for i in range(4)],[p1[i] - p2[i] for i in range(4)])

    den17=dot_product([p1[i]+p2[i]for i in range(4)], [p1[i]+p2[i]for i in range(4)])*dot_product([q17[i]-p2[i]for i in range(4)], [q17[i]-p2[i]for i in range(4)])
    M17=num17/den17
    
    #M18
    q18=[p3[i]+p4[i]for i in range(4)]
    k18=[k2[i]-q18[i]-p2[i]for i in range(4)]
    num18=2*g_x**4*dot_product([p1[i]+k1[i]for i in range(4)], [p3[i]-p4[i]for i in range(4)])
    den18=dot_product([p1[i]-k1[i]for i in range(4)], [p1[i]-k1[i]for i in range(4)])*dot_product([p3[i]+p4[i]for i in range(4)], [p3[i]+p4[i]for i in range(4)])
    M18=num18/den18
    
    #M19
    q19=[k2[i]-p4[i]for i in range(4)]
    p19=[p2[i]+p3[i]+q19[i]for i in range(4)]
    num19=g_x**3*2*dot_product([p1[i]+k1[i]for i in range(4)], [-p4[i]-k2[i]for i in range(4)])
    den19=dot_product([p1[i]-k1[i]for i in range(4)], [p1[i]-k1[i]for i in range(4)])*dot_product([p2[i]-p3[i]for i in range(4)], [p2[i]-p3[i]for i in range(4)])
    M19=num19/den19  
    
    #M20
    p20=[p4[i]-k2[i]for i in range(4)]
    q20=[p1[i]+p2[i]+p3[i]for i in range(4)]
    num20=lam_x*g_x**2*dot_product([q20[i]+k1[i]for i in range(4)], [-p4[i]-k2[i]for i in range(4)])
    den20=(dot_product([p1[i]+p2[i]+p3[i]for i in range(4)], [p1[i]+p2[i]+p3[i]for i in range(4)])-mx**2+epi)*dot_product([q20[i]-k1[i]for i in range(4)], [q20[i]-k1[i]for i in range(4)])
    M20=num20/den20
    
    #M21
    p21=[p3[i]+p4[i]-k2[i]for i in range(4)]
    q21=[p1[i]+p2[i]for i in range(4)]
    num21=lam_x*g_x**2*dot_product([p1[i]-p2[i]for i in range(4)], [k1[i]+p21[i]for i in range(4)])
    den21=dot_product([p1[i]+p2[i]for i in range(4)],[p1[i]+p2[i]for i in range(4)])*(dot_product([q21[i]-k1[i]for i in range(4)], [q21[i]-k1[i]for i in range(4)])-mx**2+epi)
    M21=num21/den21  
    
    #M22
    k22=[p4[i]-k2[i]for i in range(4)]
    p22=[p2[i]-p3[i]for i in range(4)]
    q22=[p1[i]-p22[i]for i in range(4)]
    num22=g_x**4*dot_product([p3[i]-p2[i]for i in range(4)], [p1[i]+q22[i]for i in range(4)])*dot_product([q22[i]-k1[i]for i in range(4)], [-k2[i]-p4[i]for i in range(4)])
    den22=(dot_product([p22[i]-p1[i]for i in range(4)], [p22[i]-p1[i]for i in range(4)])-mx**2+epi)*dot_product([p3[i]-p2[i]for i in range(4)], [p3[i]-p2[i]for i in range(4)])*dot_product([k2[i]-p4[i]for i in range(4)], [k2[i]-p4[i]for i in range(4)])
    M22=num22/den22
    
    #M23
    p23=[p3[i]+p4[i]for i in range(4)]
    q23=[p23[i]-k2[i]for i in range(4)]
    k23=[p1[i]+p2[i]for i in range(4)]
    num23=g_x**4*dot_product([p1[i]-p2[i]for i in range(4)], [k1[i]+q23[i]for i in range(4)])*dot_product([p3[i]-p4[i]for i in range(4)], [q23[i]+k2[i]for i in range(4)])
    den23=(dot_product([k23[i]+k1[i]-q23[i]for i in range(4)], [k23[i]+k1[i]-q23[i]for i in range(4)])-mx**2+epi)*dot_product([p1[i]+p2[i]for i in range(4)], [p1[i]+p2[i]for i in range(4)])*dot_product([p3[i]+p3[i]for i in range(4)], [p3[i]+p4[i]for i in range(4)])
    M23=num23/den23
    #M24
    q24=[p4[i]-k2[i]for i in range(4)]
    k24=[q24[i]-p3[i]for i in range(4)]
    p24=[p1[i]+p2[i]for i in range(4)]
    num24=g_x**4*dot_product([p1[i]-p2[i]for i in range(4)], [k1[i]+k24[i]for i in range(4)])*dot_product([p3[i]+k24[i]for i in range(4)], [-p4[i]-k2[i]for i in range(4)])
    den24=(dot_product([p24[i]+k1[i]-k24[i]for i in range(4)], [p24[i]+k1[i]-k24[i]for i in range(4)])-mx**2+epi)*dot_product([p1[i]+p2[i]for i in range(4)], [p1[i]+p2[i]for i in range(4)])*dot_product([p3[i]+k24[i]for i in range(4)], [p3[i]+k24[i]for i in range(4)])
    M24=num24/den24

    #M25
    k25=[p2[i]-p1[i]for i in range(4)]
    q25=[p3[i]+p4[i]+k25[i]for i in range(4)]
    num25=2*g_x**3*dot_product([p1[i]-p2[i]for i in range(4)], [k1[i]-k2[i]for i in range(4)])
    den25=(dot_product([p3[i]+p4[i]+k25[i]+q25[i]for i in range(4)], [p3[i]+p4[i]+k25[i]+q25[i]for i in range(4)]))*dot_product([p1[i]-p2[i]for i in range(4)], [p1[i]-p2[i]for i in range(4)])
    M25=num25/den25
    
    #M26
    p26=[p2[i]+p3[i]for i in range(4)]
    q26=[p4[i]-k2[i]for i in range(4)]
    num26=g_x**4*2*dot_product([p3[i]-p2[i]for i in range(4)], [-p4[i]-k2[i]for i in range(4)])
    den26=dot_product([p3[i]-p2[i]for i in range(4)], [p3[i]-p2[i]for i in range(4)]) *dot_product([p26[i]+p1[i]+k1[i]-q26[i]for i in range(4)], [p26[i]+p1[i]+k1[i]-q26[i]for i in range(4)])
    M26=num26/den26
    tot_M=M1+M2+M3+M4+M5+M6+M7+M8+M9+M10+M11+M12+M13+M14+M15+M16+M17+M18+M19+M20+M21+M22+M23+M24+M25+M26
    return tot_M
#matrix for radtive

def RadPertMat(p1,p2,p3,p4):

  P1=[0]*4;Q1a=[0]*4;Q1b=[0]*4;P2=[0]*4;
  P3=[0]*4;t31=[0]*4;t32=[0]*4;d31=[0]*4;d32=[0]*4
  Q3=[0]*4;Q4=[0]*4;k5=[0]*4;q=[0]*4;P6=[0]*4;q6=[0]*4
  P7=[0]*4;q7=[0]*4;k8=[0]*4;P8=[0]*4;q8=[0]*4
  p9=[0]*4;k9=[0]*4;q9=[0]*4
  p10=[0]*4;k10=[0]*4;q10=[0]*4
  p11=[0]*4;q11=[0]*4
  P12=[0]*4;Q12=[0]*4
  Q13=[0]*4;P13=[0]*4
  p14=[0]*4;k14=[0]*4;q14=[0]*4
  p15=[0]*4;k15=[0]*4;q15=[0]*4
  p16=[0]*4;k16=[0]*4;q16=[0]*4
  p17=[0]*4;q17=[0]*4
  q18=[0]*4;k18=[0]*4
  q19=[0]*4;p19=[0]*4
  p20=[0]*4;q20=[0]*4;p21=[0]*4;q21=[0]*4
  k22=[0]*4;p22=[0]*4;q22=[0]*4
  p23=[0]*4;q23=[0]*4;k23=[0]*4
  k24=[0]*4;p24=[0]*4;q24=[0]*4
  k25=[0]*4;q25=[0]*4;p26=[0]*4;q26=[0]*4;q5=[0]*4
  
  for i in range(4):
    #Momentum Definitions
    P1[i]=p1[i]+p2[i]+p3[i]
    P2[i]=p1[i]+p2[i]
    P3[i]=p1[i]  
    t31[i]=p2[i]+P3[i] 
    t32[i]=-p3[i]-p4[i] 
    d31[i]=p2[i]-P3[i] 
    d32[i]=P3[i]+p1[i] 
    Q3[i]=P3[i]+p1[i]
    Q4[i]=p1[i]+p2[i]-p2[i]
    k5[i]=p4[i]
    q5[i]=p3[i]-k5[i]
    P6[i]=p3[i]+p2[i]+p4[i]
    q6[i]=P6[i]+p1[i]
    P7[i]=p3[i]+p4[i]
    q7[i]=p2[i]-P7[i]
    k8[i]=p3[i]-p4[i]
    q8[i]=p2[i]-k8[i]
    P8[i]=p1[i]+q8[i]
    p9[i]=p3[i]+p4[i]
    k9[i]=p9[i]
    q9[i]=p2[i]-k9[i]
    k10[i]=p4[i]
    q10[i]=p3[i]-k10[i]
    p10[i]=p2[i]-q10[i]
    p11[i]=p2[i]+p3[i]+p4[i]
    q11[i]=p11[i]+p1[i]
    P12[i]=p1[i]
    Q12[i]=p2[i]+p3[i]+p4[i]
    Q13[i]=p3[i]+p4[i]
    P13[i]=p2[i]-Q13[i]
    k14[i]=p3[i]-p4[i]
    p14[i]=p2[i]-k14[i]
    q14[i]=p1[i]+p14[i]
    p15[i]=p3[i]+p4[i]
    k15[i]=p15[i]
    q15[i]=p2[i]-k15[i]
    q16[i]=p4[i]
    k16[i]=p3[i]-q16[i]
    p16[i]=p2[i]-k16[i]
    p17[i]=p3[i]-p4[i]
    q17[i]=p2[i]+p1[i]-p17[i]
    q18[i]=p3[i]+p4[i]
    k18[i]=-q18[i]-p2[i]
    q19[i]=-p4[i]
    p19[i]=p2[i]+p3[i]+q19[i]
    
    p20[i]=p4[i]
    q20[i]=p1[i]+p2[i]+p3[i]
    p21[i]=p3[i]+p4[i]
    q21[i]=p1[i]+p2[i]
    k22[i]=p4[i]
    p22[i]=p2[i]-p3[i]
    q22[i]=p1[i]-p22[i]
    p23[i]=p3[i]+p4[i]
    q23[i]=p23[i]
    k23[i]=p1[i]+p2[i]
    q24[i]=p4[i]
    k24[i]=q24[i]-p3[i]
    p24[i]=p1[i]+p2[i]
    k25[i]=p2[i]-p1[i]
    q25[i]=p3[i]+p4[i]+k25[i]
    p26[i]=p2[i]+p3[i]
    q26[i]=p4[i]
    #1
    num1=(dot_product(k10, p10)-dot_product(k10, p2)+dot_product(p10, p3)-dot_product(p2, p3))*(dot_product(p3,p4)+dot_product(p4, q10))*2*g_x**8
    den1=dot_product(p1, p1)*(dot_product([p10[j]-p2[j] for j in range(4)], [p10[j]-p2[j]for j in range(4)])-mx**2+epi)*dot_product([p3[j]-q10[j]for j in range(4)], [p3[j]-q10[j]for j in range(4)])
    #2
    num2=2*g_x**8*((dot_product(p1, q10)-dot_product(p1,p2))*(dot_product(p3, p4)+dot_product(p4, q10)))
    den2=dot_product(p1, p1)*(dot_product([p10[j]-p2[j] for j in range(4)], [p10[j]-p2[j]for j in range(4)])-mx**2+epi)*dot_product([p3[j]-q10[j]for j in range(4)], [p3[j]-q10[j]for j in range(4)])
    #3
    num3=num2
    den3=den2
    #4
    num4=2*g_x**8*((dot_product(p1, q10)-dot_product(p1,p2))*(dot_product(k10, p10)-dot_product(k10, p2)+dot_product(p10, p3)-dot_product(p2, p3)))
    den4=den3
    #5
    num5=2*g_x**8*((dot_product(k9, p3)-dot_product(k9,p4))*(dot_product(p9, q9)+dot_product(p2, q9)))
    den5=dot_product(p1, p1)*(dot_product([p2[j]-q9[j] for j in range(4)], [p2[j]-q9[j]for j in range(4)])-mx**2+epi)*dot_product([p3[j]+p4[j]for j in range(4)], [p3[j]+p4[j]for j in range(4)])
    #6
    num6=2*g_x**8*((dot_product(p9, q9)-dot_product(p2,p9)))
    den6=den5
    #7
    num7=2*g_x**8*((dot_product(k9, p1)-dot_product(p1,p2))*(dot_product(p9, q9)+dot_product(p2, p9)))
    den7=den6
    #8
    num8=2*g_x**8*((dot_product(k10, p10)-dot_product(k10,p2)+dot_product(p10,p3)-dot_product(p3,p2))*(dot_product(p3, p4)+dot_product(q10, p4)))
    den8=den2
    #9
    num9=2*g_x**8*((dot_product(p1, q10)-dot_product(p1,p2)+dot_product(p10,k10)-dot_product(k10,p2) -dot_product(p3, p2)))
    den9=den8
    #10
    num10=8*g_x**6
    den10=(dot_product([p1[j]+p2[j] for j in range(4)], [p1[j]+p2[j]for j in range(4)]))*dot_product([p3[j]+p4[j]+k25[j]+q25[j]for j in range(4)], [p3[j]+p4[j]+k25[j]+q25[j]for j in range(4)])
    #11
    num11=4*g_x**6
    den11=(dot_product([p2[j]+p3[j] for j in range(4)], [p2[j]+p3[j]for j in range(4)]))*dot_product([p1[j]+p26[j]-q26[j] for j in range(4)], [p1[j]+p26[j]-q26[j] for j in range(4)])
    #12
    num12=4*g_x**6*lam_x
    den12=dot_product([p1[j]+p2[j]for j in range(4)], [p1[j]+p2[j]for j in range(4)])*dot_product([q17[j]-p2[j]for j in range(4)], [q17[j]-p2[j]for j in range(4)])
    #13
    num13=2*g_x**6*((dot_product(k15, p2)-dot_product(k15,p4)))
    den13=dot_product(p1, p1)*(dot_product([p2[j]-q15[j] for j in range(4)], [p2[j]-q15[j]for j in range(4)])-mx**2+epi)*dot_product([p2[j]+p4[j]for j in range(4)], [p2[j]+p4[j]for j in range(4)])
    #14
    num14=4*g_x**6
    den14=dot_product(p1, p1)*dot_product([p2[j]-p3[j]for j in range(4)], [p2[j]-p3[j]for j in range(4)])
    #15
    num15=num14
    den15=dot_product(p1, p1)*dot_product([p4[j]+p3[j]for j in range(4)], [p4[j]+p3[j]for j in range(4)])
    #16
    num16=num13
    den16=den13
    #17
    num17=4*g_x**6*((-dot_product(k24, p4)-dot_product(p3,p4)))
    den17= dot_product([p1[j]+p2[j] for j in range(4)], [p1[j]+p2[j] for j in range(4)]) \
       * (dot_product([k24[j]-p24[j] for j in range(4)], [k24[j]-p24[j] for j in range(4)]) - mx**2 + epi) \
       * dot_product([p3[j]-k24[j] for j in range(4)], [p3[j]-k24[j] for j in range(4)])

   #18
    num18=4*g_x**6*((dot_product(p3, p4)-dot_product(k16,p4)))
    den18=dot_product(p1, p1)*(dot_product([p2[j]-p16[j] for j in range(4)], [p2[j]-p16[j]for j in range(4)])-mx**2+epi)*dot_product([k16[j]+p3[j]for j in range(4)], [k16[j]+p3[j]for j in range(4)])
    #19
    num19=num18/2
    den19=den18
    #20
    num20=2*g_x**6*((dot_product(k16, p1)-dot_product(p1,p2)))
    den20=den19
    #21
    num21=2*g_x**6*((dot_product(k24, p1)-dot_product(k24,p2)))
    den21= den17
    #22
    num22=2*g_x**6 *((dot_product(k15, p1)-dot_product(p1,p2)))
    den22=dot_product(p1, p1)*(dot_product([p2[j]-q15[j] for j in range(4)], [p2[j]-q15[j]for j in range(4)])-mx**2+epi)*dot_product([p2[j]+p4[j]for j in range(4)], [p2[j]+p4[j]for j in range(4)])
    #23
    num23=2*g_x**6 *((dot_product(q23, p1)-dot_product(q23,p2)))
    den23= dot_product([p1[j]+p2[j] for j in range(4)], [p1[j]+p2[j] for j in range(4)]) \
       * (dot_product(k23 ,k23) - mx**2 + epi) \
       * dot_product([p3[j]+p4[j] for j in range(4)], [p3[j]+p4[j] for j in range(4)])
    #24
    num24=2*g_x**6 *((dot_product(p24, p1)-dot_product(p24,p2)))
    den24=dot_product([p1[j]+p2[j] for j in range(4)], [p1[j]+p2[j] for j in range(4)]) \
       * (dot_product([k24[j]-p24[j] for j in range(4)], [k24[j]-p24[j] for j in range(4)]) - mx**2 + epi) \
       * dot_product([p3[j]-k24[j] for j in range(4)], [p3[j]-k24[j] for j in range(4)])
    #25
    num25=2*g_x**6*((-dot_product(p1, p2)+dot_product(p1,p3)+dot_product(p2,q22)-dot_product(p3,q22)))
    den25=dot_product([p3[j]-p2[j] for j in range(4)], [p3[j]-p2[j] for j in range(4)]) \
       * (dot_product([p22[j]-p1[j] for j in range(4)], [p22[j]-p1[j] for j in range(4)]) - mx**2 + epi) \
       * dot_product(p4,p4)
    #26
    num26=4*g_x**6
    den26=dot_product(p1, p1)*dot_product([p2[j]-p3[j]for j in range(4)], [p2[j]-p3[j]for j in range(4)])
    #27
    num27=num25
    den27=den25
    #28
    num28=2*g_x**6 *((dot_product(q23, p2)-dot_product(q23,p1)))
    den28=dot_product([p1[j]+p2[j] for j in range(4)], [p1[j]+p2[j] for j in range(4)]) \
       * (dot_product(k23, k23)- mx**2 + epi) \
       * dot_product([p3[j]+p4[j] for j in range(4)], [p3[j]+p4[j] for j in range(4)])
    #29
    num29=2*g_x**6*((-dot_product(p2, p3)+dot_product(p2,p4)+dot_product(p3,q8)-dot_product(p4,q8)))
    den29=dot_product([p1[j]+q8[j] for j in range(4)], [p1[j]+q8[j] for j in range(4)]) \
       * (dot_product([p1[j]-q8[j] for j in range(4)], [p1[j]-q8[j] for j in range(4)]) - mx**2 + epi) \
       * dot_product([p3[j]+p4[j] for j in range(4)], [p3[j]+p4[j] for j in range(4)])
    
    #30
    num30=4*lam_x*g_x**5
    den30=(dot_product([p3[j]-q5[j] for j in range(4)], [p3[j]-q5[j] for j in range(4)]))*(dot_product(p2 ,p2) - mx**2 + epi)
    #31
    num31=2*lam_x*g_x**5
    den31=dot_product([p1[j]+P6[j] for j in range(4)], [p1[j]+P6[j] for j in range(4)]) \
       * (dot_product([p1[j]-P6[j] for j in range(4)], [p1[j]-P6[j] for j in range(4)]) - mx**2 + epi) 
    #32
    num32=4*lam_x*g_x**5
    den32=dot_product([p3[j]-q5[j] for j in range(4)], [p3[j]-q5[j] for j in range(4)]) \
       * (dot_product(p2,p2) - mx**2 + epi) 
    #33
    num33=4*lam_x*g_x**4
    den33=dot_product(p1,p1) \
       * (dot_product(P12,P12) - mx**2 + epi) 
    #34
    num34=4*lam_x*g_x**4
    den34=dot_product(p1,p1) \
       * (dot_product([P13[j]-p2[j] for j in range(4)], [P13[j]-p2[j] for j in range(4)]) - mx**2 + epi) 
    #35
    num35=4*lam_x*g_x**4
    den35=dot_product(p1,p1) \
       * (dot_product([p2[j]-q7[j] for j in range(4)], [p2[j]-q7[j] for j in range(4)]) - mx**2 + epi) 
    #36
    num36=2*lam_x*g_x**4
    den36=(dot_product([p2[j]+p1[j] for j in range(4)], [p2[j]+p1[j] for j in range(4)])) \
       * (dot_product(q21,q21) - mx**2 + epi)  
    #37
    num37=2*lam_x*g_x**4
    den37=(dot_product([p3[j]+p4[j] for j in range(4)], [p3[j]+p4[j] for j in range(4)])) \
       * (dot_product(p2,p2) - mx**2 + epi)  
    #38
    num38=4*lam_x*g_x**4
    den38=dot_product(q20,q20) \
       * (dot_product([p1[j]+p2[j]+p3[j] for j in range(4)], [p1[j]+p2[j]+p3[j] for j in range(4)]) - mx**2 + epi) 
    #39
    num39=4*lam_x*g_x**4
    den39=dot_product(p1,p1) \
       * (dot_product([p1[j]-q7[j] for j in range(4)], [p1[j]-q7[j] for j in range(4)]) - mx**2 + epi) 
    #40
    num40=4*lam_x*g_x**4
    den40=dot_product(p1,p1) \
       * (dot_product([p2[j]-q7[j] for j in range(4)], [p2[j]-q7[j] for j in range(4)]) - mx**2 + epi)
    #41
    num41=4*lam_x*g_x**4
    den41=(dot_product([p1[j]-q11[j] for j in range(4)], [p1[j]-q11[j] for j in range(4)])) \
       * (dot_product([p1[j]+q11[j] for j in range(4)], [p1[j]+q11[j] for j in range(4)]) - mx**2 + epi)
    #42
    num42=4*lam_x*g_x**4
    den42=dot_product(p1,p1) \
       *(dot_product(P12,P12) - mx**2 + epi) 
    #43
    num43=4*lam_x*g_x**4
    den43=dot_product(q20,q20) \
       * (dot_product([p1[j]+p2[j]+p3[j] for j in range(4)], [p1[j]+p2[j]+p3[j] for j in range(4)]) - mx**2 + epi) 
    #44
    num44=g_x**3*lam_x*((dot_product(p1, P7)-dot_product(p2,p1)))
    den44=dot_product(p1,p1) \
       * (dot_product([p2[j]-q7[j] for j in range(4)], [p2[j]-q7[j] for j in range(4)]) - mx**2 + epi) * (dot_product(q7,q7) - mx**2 + epi) 
    #45
    num45=lam_x*g_x**2*((dot_product(p1, Q12)))
    den45=(dot_product(Q12,Q12) - mx**2 + epi)*dot_product(p1,p1)
    
    
  M1 = np.zeros(4,dtype=complex)
  M2= np.zeros(4,dtype=complex)
  M3= np.zeros(4,dtype=complex)
  M4=np.zeros(4,dtype=complex)
  M5=np.zeros(4,dtype=complex)
  M6=np.zeros(4,dtype=complex)
  M7=np.zeros(4,dtype=complex)
  M8=np.zeros(4,dtype=complex)
  M9=np.zeros(4,dtype=complex)
  M10=np.zeros(4,dtype=complex)
  M11=np.zeros(4,dtype=complex)
  M12=np.zeros(4,dtype=complex)
  M13=np.zeros(4,dtype=complex)
  M14=np.zeros(4,dtype=complex)
  M15=np.zeros(4,dtype=complex)
  M16=np.zeros(4,dtype=complex)
  M17=np.zeros(4,dtype=complex)
  M18=np.zeros(4,dtype=complex)
  M19=np.zeros(4,dtype=complex)
  M20=np.zeros(4,dtype=complex)
  M21=np.zeros(4,dtype=complex)
  M22=np.zeros(4,dtype=complex)
  M23=np.zeros(4,dtype=complex)
  M24=np.zeros(4,dtype=complex)
  M25=np.zeros(4,dtype=complex)
  M26=np.zeros(4,dtype=complex)
  M27=np.zeros(4,dtype=complex)
  M28=np.zeros(4,dtype=complex)
  M29=np.zeros(4,dtype=complex)
  M30=np.zeros(4,dtype=complex)
  M31=np.zeros(4,dtype=complex)
  M32=np.zeros(4,dtype=complex)
  M33=np.zeros(4,dtype=complex)
  M34=np.zeros(4,dtype=complex)
  M35=np.zeros(4,dtype=complex)
  M36=np.zeros(4,dtype=complex)
  M37=np.zeros(4,dtype=complex)
  M38=np.zeros(4,dtype=complex)
  M39=np.zeros(4,dtype=complex)
  M40=np.zeros(4,dtype=complex)
  M41=np.zeros(4,dtype=complex)
  M42=np.zeros(4,dtype=complex)
  M43=np.zeros(4,dtype=complex)
  M44=np.zeros(4,dtype=complex)
  M45=np.zeros(4,dtype=complex)
  Tot_mat= np.zeros(4,dtype=complex)
  for mu in range(4):  
      M1[mu] = num1*p1[mu]/den1
      M2[mu]=((p10[mu]-p2[mu])*num2)/den2
      M3[mu]=((k10[mu]+p3[mu])*num3)/den3
      M4[mu]=(p4[mu]*num4)/den4
      M5[mu]=(p1[mu]*num5)/den5
      M6[mu]=(((p3[mu]-p4[mu])*(k9[mu]-p2[mu]))*num6)/den6
      M7[mu]=((p3[mu]-p4[mu])*num7)/den7
      M8[mu]=((q10[mu]-p2[mu])*num8)/den8
      M9[mu]=((q10[mu]+p3[mu])*num9)/den9
      M10[mu]=((p1[mu]-p2[mu])*num10)/den10
      M11[mu]=((p3[mu]-p2[mu])*num11)/den11
      M12[mu]=(q17[mu]*(p3[mu]-p4[mu])*num12)/den12
      M13[mu]=(p1[mu]*num13)/den13
      M14[mu]=(p1[mu]*num14)/den14
      M15[mu]=(p1[mu]*num15)/den15
      M16[mu]=((k15[mu]-p2[mu])*num16)/den16
      M17[mu]= ((p1[mu]-p2[mu])*num17)/den17
      M18[mu]=(p1[mu]*num18)/den18
      M19[mu]=((k16[mu]-p2[mu])*num19)/den19
      M20[mu]=((k16[mu]-p3[mu])*num20)/den20
      M21[mu]=(k24[mu]+p3[mu]*num21)/(den21)
      M22[mu]=(p2[mu]-p4[mu]*num22)/(den22)
      M23[mu]=(p3[mu]-p4[mu]*num23)/(den23)
      M24[mu]=(p4[mu]*num24)/den24
      M25[mu]=(p4[mu]*num25)/den25
      M26[mu]=(p4[mu]*num26)/den26
      M27[mu]=(q22[mu]*num27)/den27
      M28[mu]=(q23[mu]*num28)/den28
      M29[mu]=(p1[mu]-q8[mu]*num29)/den29
      M30[mu]=(p4[mu]*num30)/den30
      M31[mu]=(p1[mu]+P6[mu]*num31)/den31
      M32[mu]=(p3[mu]+q5[mu]*num32)/den32
      M33[mu]=(p1[mu]*num33)/den33
      M34[mu]=(p1[mu]*num34)/den34
      M35[mu]=(p1[mu]*num35)/den35
      M36[mu]=(p21[mu]*num36)/den36
      M37[mu]=(p3[mu]-p4[mu]*num37)/den37
      M38[mu]=(p4[mu]*num38)/den38
      M39[mu]=(p4[mu]*num39)/den39
      M40[mu]=(P7[mu]-p2[mu]*num40)/den40
      M41[mu]=(p1[mu]+q11[mu]*num41)/den41
      M42[mu]=(Q12[mu]*num42)/den42
      M43[mu]=(q20[mu]*num43)/den43
      M44[mu]=(P7[mu]*num44/den44)
      M45[mu]=(p1[mu]*num44/den44)
      
      
      Tot_mat[mu]=M1[mu]+M2[mu]+M3[mu]+M4[mu]+M5[mu]+M6[mu]+M7[mu]+M8[mu]+M9[mu]+M10[mu]+M11[mu]+M12[mu]+M13[mu]+M14[mu]+M15[mu]+M16[mu]+M17[mu]+M18[mu]+M19[mu]+M20[mu]+M21[mu]+M22[mu]+M23[mu]+M24[mu]+M25[mu]+M26[mu]+M27[mu]+M28[mu]+M29[mu]+M30[mu]+M31[mu]+M32[mu]+M33[mu]+M34[mu]+M35[mu]+M36[mu]+M37[mu]+M38[mu]+M39[mu]+M40[mu]+M41[mu]+M42[mu]+M43[mu]+M44[mu]+M45[mu]
  return Tot_mat
#atrix element for process 1
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


#matrix elements for process 2
def MatC_D(gx, p1, p2, q):
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

#matrix elements for process 5 

def MatC_5(p1_5,p2_5,p3_5,p4_5):
   # Denominator terms
   denom1 = dot_product(p3_5, p3_5)- 2*dot_product(p1_5, p3_5)
   denom2 =  dot_product([p1_5[i] - p3_5[i] for i in range(4)], [p1_5[i] - p3_5[i] for i in range(4)])
   denom3 = dot_product(p3_5, p3_5)- 2*dot_product(p2_5, p3_5)
  
   
   V = np.zeros(4)
   for mu in range(4):
       Num1=g_x**2*(p3_5[mu]-2*p1_5[mu])*(p4[mu]-2*p2[mu])
       Num2=g_x**2*(2*p1_5[mu]-2*p3_5[mu])*(p4[mu]-2*p2[mu])
       Num3=g_x**2(p3_5[mu]-2*p2_5[mu])*(p4[mu]-2*p1[mu])
       Num4=2*g_x**2
       V[mu] = gx**3 * (
   return V

def MatC_6(p1_5,p2_5,p3_5,p4_5):
   # Denominator terms
   denom1 = dot_product(p1, p1)
   denom2 =  dot_product([p1_5[i] - p3_5[i] for i in range(4)], [p1_5[i] - p3_5[i] for i in range(4)])
   denom3 =  dot_product([p1[i] - q[i] for i in range(4)], [p1[i] - q[i] for i in range(4)])
   Denom=denom1*denom2*denom3
   
   V = np.zeros(4)
   for mu in range(4):
       Num1=2*dot_product(p1, p1)*p2[mu]*(mx**2-dot_product(q, q))
       Num2=2*p1[mu]*dot_product([p1[i] - q[i] for i in range(4)], [p1[i] - q[i] for i in range(4)])*(dot_product(q, q)- mx**2)
       Num=Num1+Num2
       V[mu] = gx**3 * (Num/Denom)
   return V
if __name__ == "__main__":
    v_rel_km_s=300 #km/s
    beta_rel=v_rel_km_s/2.99e8
    s=4*4*mx**2
    V0 = g_x**2
    mu = mx / 2
    a0=1/(mu*V0)
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
    
    # In CM frame: 
    #incomming
    E = mx
    p_mag = mx * beta_rel / 2
    p1 = np.array([E, p_mag, 0, 0])   # Four-vector for particle 1
    p2 = np.array([E, -p_mag, 0, 0])  # Four-vector for particle 2
    p3 = np.array([E, 0, p_mag, 0])
    p4 = np.array([E, 0, -p_mag, 0])
    M_rad=RadPertMat(p1,p2,p3,p4)
    #mat sq for raditive proces
    
    def PertMatC(M):
           # Extract spatial components (j = 1,2,3)
           Mj = np.array(M[1:4])   # [Mx, My, Mz]
           # Compute spatial dot products
           Mj_dot_Mj = np.abs(np.dot(Mj, Mj))          # Σ_j M^j M^j*
           # Apply the polarization sum formula
           return Mj_dot_Mj
    RadMat=PertMatC(M_rad)
    #outoging
    Ef=np.sqrt(s)/2
    k_mag=0.5*np.sqrt(s-4*mx**2)
    def M_sq(theta,phi):# for normal process
        E = mx
        p_mag = mx * beta_rel / 2
        p1 = np.array([E, p_mag, 0, 0])   # Four-vector for particle 1
        p2 = np.array([E, -p_mag, 0, 0])  # Four-vector for particle 2
        p3 = np.array([E, 0, p_mag, 0])
        p4 = np.array([E, 0, -p_mag, 0])
        k1=np.array([Ef,k_mag*np.sin(theta)*np.cos(phi),k_mag*np.sin(theta)*np.sin(phi),k_mag*np.cos(theta)])
        k2=np.array([Ef,-k_mag*np.sin(theta)*np.cos(phi),-k_mag*np.sin(theta)*np.sin(phi),-k_mag*np.cos(theta)])
        M_free=pertmat(p1,p2,p3,p4,k1,k2)
       
        M_sq_free=np.abs(M_free)**2
       
        
        return M_sq_free
    # Cross section for process
    
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
            # Cross sections
            #Energy diff
    De1c=Ek-E1c
    De2c=Ek-E2c
    De3c=Ek-E3c
        
            #prefactors
    C1 = 1/(16*np.pi*mx**2)*(De1c)/(mu*v_rel_km_s)
    C2 = 1/(16*np.pi*mx**2)*(De2c)/(mu*v_rel_km_s)
    C3 = 1/(16*np.pi*mx**2)*(De3c)/(mu*v_rel_km_s)
           
            # Coulomb
    M1 = (1/np.sqrt(2*mu))*psi100c*PsiScatC
    M2 = (1/np.sqrt(2*mu))*psi200c*PsiScatC
    M3 = (1/np.sqrt(2*mu))*psi300c*PsiScatC
        #calculation for process 2
        # COM energy
    s1c=(MB1c+Wc1)**2
    s2c=(MB2c+Wc2)**2
    s3c=(MB3c+Wc3)**2
    S=4*mx**2
        #Extracting pertubative |M|^2
        #columbic
    V1cd=MatC_D(g_x,  p1, p2, q_vec1c)
    V2cd=MatC_D(g_x,  p1, p2, q_vec2c)
    V3cd=MatC_D(g_x,  p1, p2, q_vec3c)
    MP1cd=PertMatC(V1cd)
    MP2cd=PertMatC(V2cd)
    MP3cd=PertMatC(V3cd)
      
        
        
        # Cross sections 
        #prefactors
    C1d = 1/(32*np.pi*MB1c*Wc1)*np.sqrt(s1c-S)/np.sqrt(s1c)
    C2d = 1/(32*np.pi*MB2c*Wc2)*np.sqrt(s2c-S)/np.sqrt(s2c)
    C3d = 1/(32*np.pi*MB3c*Wc3)*np.sqrt(s3c-S)/np.sqrt(s3c)
    
        
        # Coulomb
    M1d = np.sqrt(1/(2*mu))*psi100c
    M2d = np.sqrt(1/(2*mu))*psi200c
    M3d = np.sqrt(1/(2*mu))*psi300c
        
      
    # integrating over solid angle
# Monte Carlo integration(only for normal process)
    N_samples = 10

    theta_random = np.random.uniform(0, np.pi, N_samples)
    phi_random = np.random.uniform(0, 2*np.pi, N_samples)

# Integrand includes sin(theta)
    M_values = np.array([M_sq(theta_random[i], phi_random[i]) * np.sin(theta_random[i]) for i in range(N_samples)])

# Monte Carlo estimate: average * total integration volume (4*pi)
    I_MC = (4 * np.pi / N_samples) * np.sum(M_values)
    Therm_cs=np.sqrt(3)/(256*mx**4*np.pi)*I_MC# got from simpler realisation of DM
# PROCESS 1
    cs1_1=(C1*np.abs(M1)**2*MP1c)
    cs2_1=(C2*np.abs(M2)**2*MP2c)
    cs3_1=(C3*np.abs(M3)**2*MP3c)
# PROCESS 2
    cs1_2=(C1d*M1d**2*MP1cd)
    cs2_2=(C2d*M2d**2*MP2cd)
    cs3_2=(C3d*M3d**2*MP3cd)
  # PROCESS 3
    cs3=np.sqrt(3)/(256*mx**4*np.pi)*I_MC
        # PROCESS 4
    s4=16*mx**2
    MatSq1_4=psi100c**2*RadMat
    MatSq2_4=psi200c**2*RadMat
    MatSq3_4=psi300c**2*RadMat
    p1_4=np.sqrt((s4+MB1c)**2-4*s4*MB1c**2)/s4
    p2_4=np.sqrt((s4+MB1c)**2-4*s4*MB1c**2)/s4
    p3_4=np.sqrt((s4+MB1c)**2-4*s4*MB1c**2)/s4
    CS1_4=(MatSq1_4*p1_4)/(64*np.pi*mx**4)
    C21_4=(MatSq2_4*p2_4)/(64*np.pi*mx**4)
    CS3_4=(MatSq3_4*p3_4)/(64*np.pi*mx**4)
    
    #PROCESS 5
    M5=2*g_x**2
    cs5=M5**2/(32*np.pi*mx**2)
    #PROCESS 6
    M6=M5
    CS6=
    
    
    #Process 7
    termc1=(4*mx**2+MB1c**2)**2/(4*mx**2)-MB1c**2
    Q_cm_c1=np.sqrt(termc1)
    termc2=(4*mx**2+MB2c**2)**2/(4*mx**2)-MB2c**2
    Q_cm_c2=np.sqrt(termc2)
    termc3=(4*mx**2+MB3c**2)**2/(4*mx**2)-MB3c**2
    Q_cm_c3=np.sqrt(termc3)
    q_vec1c = [Wc1, 0, 0, Qc1]
    q_vec2c = [Wc2, 0, 0, Qc2]
    q_vec3c = [Wc3, 0, 0, Qc3]
    V1cr=MatC(g_x, p1, p2, q_vec1c)
    V2cr=MatC(g_x, p1, p2, q_vec2c)
    V3cr=MatC(g_x, p1, p2, q_vec3c)
    MP1cR=PertMatCR(V1cr)
    MP2cR=PertMatCR(V2cr)
    MP3cR=PertMatCR(V3cr)
    C21 = 1/(16*np.pi*mx**2)*(Q_cm_c1/(2*mx))
    C32 = 1/(16*np.pi*mx**2)*(Q_cm_c2/(2*mx))
    C31 = 1/(16*np.pi*mx**2)*(Q_cm_c1/(2*mx))
    M21c = np.sqrt(2/mu)*psi100c*psi200c
    M32c = np.sqrt(2/mu)*psi200c*psi300c
    M31c = np.sqrt(2/mu)*psi300c*psi100c
    cs1c_7=(C21*M21c**2*MP1cR)
    cs2c_7=(C32*M32c**2*MP2cR)
    cs3c_7=(C31*M31c**2*MP3cR)
    
    #Process 9
    s9=4*mx**2*(2*mx)**2
    MatSq1_9=psi100c**2*I_MC
    MatSq2_9=psi200c**2*I_MC
    MatSq3_9=psi300c**2*I_MC
    P9=(s9-4*mx**2)/4
    CS1_9=(np.sqrt(s9-4*mx**2)*MatSq1_9)/(64*np.pi*mx**2*(2*mx**2)*np.sqrt(s9))   
    CS2_9=(np.sqrt(s9-4*mx**2)*MatSq2_9)/(64*np.pi*mx**2*(2*mx**2)*np.sqrt(s9)) 
    CS3_9=(np.sqrt(s9-4*mx**2)*MatSq3_9)/(64*np.pi*mx**2*(2*mx**2)*np.sqrt(s9))      
    
    #Process 10
    s10=(4*mx)**2
    MatSq12_10=psi100c*psi200c*I_MC
    MatSq13_10=psi100c*psi300c*I_MC
    MatSq11_10=psi100c*psi100c*I_MC
    MatSq23_10=psi200c*psi300c*I_MC
    MatSq22_10=psi200c*psi200c*I_MC
    MatSq33_10=psi300c*psi300c*I_MC
    
    CS11_10=(np.sqrt(s10-4*mx**2)*MatSq11_10)/(32*np.pi*(2*mx**2)**2*np.sqrt(s10))
    CS12_10=(np.sqrt(s10-4*mx**2)*MatSq12_10)/(32*np.pi*(2*mx**2)**2*np.sqrt(s10))
    CS13_10=(np.sqrt(s10-4*mx**2)*MatSq13_10)/(32*np.pi*(2*mx**2)**2*np.sqrt(s10))
    CS23_10=(np.sqrt(s10-4*mx**2)*MatSq23_10)/(32*np.pi*(2*mx**2)**2*np.sqrt(s10))
    CS22_10=(np.sqrt(s10-4*mx**2)*MatSq22_10)/(32*np.pi*(2*mx**2)**2*np.sqrt(s10))
    CS33_10=(np.sqrt(s10-4*mx**2)*MatSq33_10)/(32*np.pi*(2*mx**2)**2*np.sqrt(s10))
    # Process 11
    s11=4*mx**2*(2*mx)**2
    MatSq11_11=psi100c*psi100c*RadMat
    MatSq12_11=psi100c*psi200c*RadMat
    MatSq13_11=psi100c*psi300c*RadMat
    MatSq23_11=psi200c*psi300c*RadMat
    MatSq22_11=psi200c*psi200c*RadMat
    MatSq33_11=psi300c*psi300c*RadMat
    P11=np.sqrt((s11+(2*mx))**2-4*s11*(2*mx)**2)/s11
    CS11_11=(P11*MatSq11_11)/(32*mx*(2*mx)*np.pi)
    CS12_11=(P11*MatSq12_11)/(32*mx*(2*mx)*np.pi)
    CS13_11=(P11*MatSq13_11)/(32*mx*(2*mx)*np.pi)
    CS23_11=(P11*MatSq23_11)/(32*mx*(2*mx)*np.pi)
    CS22_11=(P11*MatSq22_11)/(32*mx*(2*mx)*np.pi)
    CS33_11=(P11*MatSq33_11)/(32*mx*(2*mx)*np.pi)
    #Process 12 
    s12=(4*mx)**2
    MatSq111_12=psi100c*psi100c*psi100c*RadMat
    MatSq112_12=psi100c*psi100c*psi200c*RadMat
    MatSq113_12=psi100c*psi100c*psi300c*RadMat
    MatSq122_12=psi100c*psi200c*psi200c*RadMat
    MatSq123_12=psi100c*psi200c*psi300c*RadMat
    MatSq133_12=psi100c*psi300c*psi300c*RadMat
    MatSq222_12=psi200c*psi200c*psi200c*RadMat
    MatSq223_12=psi200c*psi200c*psi300c*RadMat
    MatSq233_12=psi200c*psi300c*psi300c*RadMat
    MatSq333_12=psi300c*psi300c*psi300c*RadMat
    P12=np.sqrt((s12+(2*mx))**2-4*s12*(2*mx)**2)/s12
    CS111_12=(MatSq111_12*P12)/(32*(2*mx)**2*np.pi)
    CS112_12=(MatSq112_12*P12)/(32*(2*mx)**2*np.pi)
    CS113_12=(MatSq113_12*P12)/(32*(2*mx)**2*np.pi)
    CS122_12=(MatSq122_12*P12)/(32*(2*mx)**2*np.pi)
    CS123_12=(MatSq123_12*P12)/(32*(2*mx)**2*np.pi)
    CS133_12=(MatSq133_12*P12)/(32*(2*mx)**2*np.pi)
    CS222_12=(MatSq222_12*P12)/(32*(2*mx)**2*np.pi)
    CS223_12=(MatSq223_12*P12)/(32*(2*mx)**2*np.pi)
    CS233_12=(MatSq233_12*P12)/(32*(2*mx)**2*np.pi)
    CS333_12=(MatSq333_12*P12)/(32*(2*mx)**2*np.pi)
    