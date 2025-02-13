import numpy as np
from hepunits import MeV as m
from hepunits import GeV as g
from scipy.integrate import quad,simps
import matplotlib.pyplot as plt

# Initialize results list to store cross-sections for different parameter sets
results = []
bullet_M_c = []
bullet_Y_cp = []
abell_Mc=[]
abell_Ycp=[]
Unitary_M_c=[]
Unitary_Y_cp=[]

 # Function for process 1
def M(theta):
     
     M1 = -4 * L_c
     M2 = (mu_c + Y_cp * v_p)**2 / (3 * M_c**2)
     M3 = (-(L_ch * v_h * np.cos(x) - (L_cp * v_p + mu_cp) * np.sin(x)))**2 / (-2 * M_c**2 * (1 - np.cos(theta)) - m_1**2)
     M4 = (-(L_ch * v_h * np.cos(x) - (L_cp * v_p + mu_cp) * np.sin(x)))**2 / (-2 * M_c**2 * (1 + np.cos(theta)) - m_1**2)
     M5 = (-(L_ch * v_h * np.sin(x) + (L_cp * v_p + mu_cp) * np.cos(x)))**2 / (-2 * M_c**2 * (1 - np.cos(theta)) - m_2**2)
     M6 = (-(L_ch * v_h * np.sin(x) + (L_cp * v_p + mu_cp) * np.cos(x)))**2 / (-2 * M_c**2 * (1 + np.cos(theta)) - m_2**2)
     M_net_1 = (M1 + M2 + M3 + M4 + M5 + M6)
     M_net=(M_net_1)**2  *np.sin(theta)
    
     return M_net
 # Function for process 2
def Mb(theta):
     M1 = -4 * L_c
     M2 = (mu_c + Y_cp * v_p)**2 / (-2 * M_c**2 * (1 + np.cos(theta)) - M_c**2)
     M3 = (-(L_ch * v_h * np.cos(x) - (L_cp * v_p + mu_cp) * np.sin(x)))**2 / (-2 * M_c**2 * (1 - np.cos(theta)) - m_1**2)
     M4 = (-(L_ch * v_h * np.cos(x) - (L_cp * v_p + mu_cp) * np.sin(x)))**2 / (4 * M_c**2 - m_1**2)
     M5 = (-(L_ch * v_h * np.sin(x) + (L_cp * v_p + mu_cp) * np.cos(x)))**2 / (-2 * M_c**2 * (1 - np.cos(theta)) - m_2**2)
     M6 = (-(L_ch * v_h * np.sin(x) + (L_cp * v_p + mu_cp) * np.cos(x)))**2 / (4 * M_c**2 - m_2**2)
     M_net_2 = (M1 + M2 + M3 + M4 + M5 + M6)
     M_net_b=(M_net_2)**2  *np.sin(theta)
     return M_net_b
 

# Loop over M_c and Y_cp
for M_c in np.arange(0.001*g , 0.6*g , 0.001*g ):
    hbar_c = 6.582e-22 * 2.998e10  # [6.582e-22 MeV s] * [2.998e10 cm/s]
    MeV_to_g = 1.78266192e-27  # 1 MeV/c^2 = 1.78e-27 g
   # Convert masses to grams
    m_c_grams = M_c * MeV_to_g # mass in gram
    Mc_cm=M_c/hbar_c # mass in cm^-1
    
   
    L_ch = 0.001
    L_c = 1
    v_h = 246*g
    L_cp = 0.5
    v_p = 60*M_c
    mu_cp = 0.1*M_c
    x=np.pi/4
    m_1 = 25*M_c
    mu_c = 0.005*M_c
    m_2 = 125*g
    for Y_cp in np.arange(0, 0.2, 0.01):
       
        # Numerical integration for process 1
        r_numerical, error = quad(M, 0,(np.pi))
        # Numerical integration for process 2
        r_numerical_2, error2 = quad(Mb, 0,np.pi)
        cs_tot=(2/(64*np.pi*(Mc_cm)**2))*(r_numerical+r_numerical_2)       
        #cs_tot_cm = cs_tot*0.389*10**(-27)#cross section in cm^2
        #cs_b=cs_tot_cm/M_c_g# cros section in cm^2/g
        cs_b=cs_tot/(m_c_grams)# scaled cross section with no units
        cs_b_gev=cs_b*g**(-3) # scaled cross section with GeV^-3 unit
        print(cs_tot,cs_b,cs_b_gev) 
        with open('Full_data.txt', 'a') as f1:
            f1.write(f"{M_c} {Y_cp}  {cs_tot}  {cs_b} {cs_b_gev}\n")
        if  cs_b <= 1:
            bullet_M_c.append(M_c) 
            bullet_Y_cp.append(Y_cp)
            with open('Bullet_bound_in_gev.txt', 'a') as output:  # Use 'a' to append
               output.write(f"{M_c} {Y_cp} {cs_b_gev}\n")
        if 1<=cs_b<=3:
            abell_Mc.append(M_c)
            abell_Ycp.append(Y_cp)
            with open('Abell_bound_in_cm_per_g.txt', 'a') as output:  # Use 'a' to append
               output.write(f"{M_c} {Y_cp} {cs_b_gev}\n")
        
       
plt.scatter(bullet_M_c, bullet_Y_cp, s=10 ,c="blue", alpha=1, label="bullet cluster bound")
#plt.scatter(Unitary_M_c, Unitary_Y_cp, s=0.0025 ,c="green", alpha=0.3, label="unitary bound")
plt.scatter(abell_Mc, abell_Ycp, s=20 ,c="red", alpha=0.3, label="abell cluster bound")



plt.xlabel("M_c [MeV]")

plt.ylabel("Y_cp")
plt.ylim(0,0.2)
plt.xlim(0,500)

plt.title("Scatter Plot of M_c vs Y_cp")

plt.legend()

plt.grid(True)

plt.show()
      #  print(M_c,Y_cp) 
#print(bullet_M_c,bullet_Y_cp)

        
