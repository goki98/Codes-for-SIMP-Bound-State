import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from hepunits import MeV as M
from scipy.integrate import quad
from hepunits import GeV as G

rd=[]
mdm=[]
ycp=[]
# Function for y_eq(x1)
def y_eq(x1):
    return 0.145 * (g_dm / gs) * x1**(3/2) * np.exp(-x1)

# Function for the boltzman equation
def dydx(x1, y1):
    eq = y_eq(x1)
    return (-0.116 * gs**(3/2) * MPl * M_c**4 * Tcs * (y1**3 - y1**2 * eq)) / (x1**5)

# Backward Euler Method for solving boltzman eqn
def backward_euler(f, x0, y0, x_end, h):
    x_values = [x0]
    y_values = [y0]
    x = x0
    y = y0

    while x < x_end:
        # Define the equation to solve
        def equation(y_new):
            return y_new - y - h * f(x + h, y_new)

    # Solve for y_new using fsolve (Newton's method)
        y_new = fsolve(equation, y)[0]

    # Update x and y values
        x += h
        x_values.append(x)
        y_values.append(y_new)
        y = y_new

    return np.array(x_values), np.array(y_values)

 # Function for process 1
def M(theta):
    #Mandelstam variables
    s=9*M_c**2
    t=-4.5 * M_c**2 * (1 - np.cos(theta)) 
    u=-4.5 * M_c**2 * (1 + np.cos(theta))
    #Matrix elemnts
    M1a = (-L_a1 * -L_a2) / (s - m_1**2)
    M2a = (-L_a1 * -L_a2) / (t - m_1**2)
    M1b = (-L_b1 * -L_b2) / (s - m_2**2)
    M2b = (-L_b1 * -L_b2) / (t - m_2**2)
    M3a = ((-L_a1)**2 * (-L3)) / ((s - m_1**2) * (t - M_c**2))
    M4a = ((-L_a1)**2 * (-L3))/ ((t - m_1**2) * (s - M_c**2))
    M5a = ((-L_a1)**2 * (-L3)) / ((t - m_1**2) * (t - M_c**2))
    M3b = ((-L_b1)**2 * (-L3)) / ((s - m_2**2) * (t - M_c**2))
    M4b =  ((-L_b1)**2 * (-L3)) / ((t - m_2**2) * (s - M_c**2))
    M5b =  ((-L_b1)**2 * (-L3))/ ((t - m_2**2) * (t - M_c**2))
    M6 = (-L3 * -L4) / (s - M_c**2)
    M7 = (-L3 * -L4) / (t - M_c**2)
    M8 = (-L3)**3 / ((s - M_c**2) * (t - M_c**2))

    M_net_1= M1a + M2a + M3a + M4a + M5a + M1b + M2b + M3b + M4b + M5b +M6 + M7 + M8
    M_net=(M_net_1)**2  *np.sin(theta)
    
    return M_net
 # Function for process 2
def Mb(theta):
     #Mandelstam variables
     s=9*M_c**2
     t=-4.5 * M_c**2 * (1 - np.cos(theta)) 
     u=-4.5 * M_c**2 * (1 + np.cos(theta))
     #Matrix elements
     M11 = (-L3 * -L4) / (s - M_c**2)
     M12 = (-L3)**3 / ((s - M_c**2) * (t - M_c**2))
     M13 = (-L3 * -L4) / (t - M_c**2)
     M14 = (-L3 * -L4) / (s - M_c**2)
     M15t = (-L3)**3 / ((t - M_c**2) * (t - M_c**2))
     M15u = (-L3)**3  / ((u - M_c**2) * (t - M_c**2))
     M16t = (-L3 * -L4) / (t - M_c**2)
     M16u = (-L3 * -L4) / (u - M_c**2)
     
     M1at = (-L_a1 * -L_a2) / (t - m_1**2)
     M1au =(-L_a1 * -L_a2) / (u - m_1**2)
     M2at =(-L_a1 * -L_a2) / (t - m_1**2)
     M2au =(-L_a1 * -L_a2) / (u - m_1**2)
     M3a =(-L_a1 * -L_a2) / (t - m_1**2)
     M1bt =(-L_a1 * -L_a2) / (t - m_2**2)
     M1bu =(-L_a1 * -L_a2) / (u - m_2**2)
     M2bt =(-L_a1 * -L_a2) / (t - m_2**2)
     M2bu =(-L_a1 * -L_a2) / (u - m_2**2)
     M3b =(-L_a1 * -L_a2) / (t - m_2**2)
     
     M4at = ((-L_a1)**2 * (-L3)) / ((t - m_1**2) * (s - M_c**2))
     M4au = ((-L_a1)**2 * (-L3)) / ((u - m_1**2) * (s - M_c**2))
     M5at = ((-L_a1)**2 * (-L3)) / ((s - m_1**2) * (t - M_c**2))
     M5au = ((-L_a1)**2 * (-L3)) / ((s - m_1**2) * (u - M_c**2))
     M6at = ((-L_a1)**2 * (-L3)) / ((t - m_1**2) * (s - M_c**2))
     M6au = ((-L_a1)**2 * (-L3)) / ((u - m_1**2) * (s - M_c**2))
     M7at = ((-L_a1)**2 * (-L3)) / ((t - m_1**2) * (t - M_c**2))
     M7au = ((-L_a1)**2 * (-L3)) / ((s - m_1**2) * (u - M_c**2))
     M8a = ((-L_a1)**2 * (-L3)) / ((t - m_1**2) * (s - M_c**2))
     M9at = ((-L_a1)**2 * (-L3)) / ((t - m_1**2) * (t - M_c**2))
     M9au = ((-L_a1)**2 * (-L3)) / ((s - m_1**2) * (u - M_c**2))
     M10at = ((-L_a1)**2 * (-L3)) / ((s - m_1**2) * (t - M_c**2))
     M10au = ((-L_a1)**2 * (-L3)) / ((s - m_1**2) * (u - M_c**2))
     
     M4bt = ((-L_b1)**2 * (-L3)) / ((t - m_2**2) * (s - M_c**2))
     M4bu = ((-L_b1)**2 * (-L3)) / ((u - m_2**2) * (s - M_c**2))
     M5bt = ((-L_b1)**2 * (-L3)) / ((s - m_2**2) * (t - M_c**2))
     M5bu = ((-L_b1)**2 * (-L3)) / ((s - m_2**2) * (u - M_c**2))
     M6bt = ((-L_b1)**2 * (-L3)) / ((t - m_2**2) * (s - M_c**2))
     M6bu = ((-L_b1)**2 * (-L3)) / ((u - m_2**2) * (s - M_c**2))
     M7bt = ((-L_b1)**2 * (-L3)) / ((t - m_2**2) * (t - M_c**2))
     M7bu = ((-L_b1)**2 * (-L3)) / ((u - m_2**2) * (t - M_c**2))
     M8b = ((-L_b1)**2 * (-L3)) / ((t - m_2**2) * (s - M_c**2))
     M9bt = ((-L_b1)**2 * (-L3)) / ((t - m_2**2) * (t - M_c**2))
     M9bu = ((-L_b1)**2 * (-L3)) / ((u - m_2**2) * (t - M_c**2))
     M10bt = ((-L_b1)**2 * (-L3)) / ((s - m_2**2) * (t - M_c**2))
     M10bu = ((-L_b1)**2 * (-L3)) / ((s - m_2**2) * (u - M_c**2))
     
     M_Net_2 =  M11 + M12 + M13 + M14 + M15t + M15u + M16t + M16u + M1at + M1au + M2at + M2au + M3a + M1bt + M1bu + M2bt + M2bu + M3b + M4at + M4au + M5at + M5au + M6at + M6au + M7at + M7au + M8a + M9at + M9au + M10at + M10au + M4bt + M4bu + M5bt + M5bu + M6bt + M6bu + M7bt + M7bu + M8b + M9bt + M9bu + M10bt + M10bu


     M_net_b=(M_Net_2)**2  *np.sin(theta)
     return M_net_b
for M_c in  np.arange(0.1 , 0.6 , 0.01 ):
    for Y_cp in np.arange(0, 0.2, 0.0001):
        #Constants
        L_ch = 0.1
        L_c = 1
        v_h = 246
        L_cp = 0.0001
        v_p = 60*M_c
        mu_cp = 0.1*M_c
        c = 0.1
        s = 0.99
        m_1 = 25
        mu_c = 0.005*M_c
        m_2 = 125
        g_dm=1
        gs=10.75
        MPl = 1.22e19  # Planck mass in GeV
        x=81.99
        #Vertex functions
        L_a1=-(L_ch * v_h * np.cos(x) - (L_cp * v_p + mu_cp) * np.sin(x))
        L_a2=np.sin(x)*Y_cp
        L_b1=-(L_ch * v_h * np.sin(x) + (L_cp * v_p + mu_cp) * np.cos(x))
        L_b2=np.cos(x)*Y_cp
        L3=(mu_c + Y_cp * v_p)
        L4=-4 * L_c
       
        
        #3 to 2 thermal avg cross section
        r_numerical, error = quad(M, 0,np.pi)
        mat1=r_numerical      
        r_numerical_2, error2 = quad(Mb, 0,np.pi)
        mat2=r_numerical_2
        M2=r_numerical+r_numerical_2
        Tcs=(np.sqrt(5)*2*M2)/(192*np.pi*M_c**3)

        # Initial conditions
        x0 = 1.0  # initial x1 value
        y0 = 0.1  # initial y1 value
        x_end = 1000.0  # final x1 value
        h = 0.1  # step size
        x_vals_backward, y_vals_backward = backward_euler(dydx, x0, y0, x_end, h)
        #Yield at x tends to infinity i.e yield after freeze out 
        Y_f=y_vals_backward[-1]
        #print(Tcs,Y_f)
   #    relic density calculation
        const1=2.752*10**8
        const2=(M_c)
        omega=const1*const2*Y_f
        print(M_c,omega,Tcs,Y_f)
        if 0.11<=omega<=0.13:
            rd.append(omega)
            mdm.append(M_c)
            ycp.append(Y_cp)
            with open('RelicDensity_parametr_space_V2_run4.txt', 'a') as output:  # Use 'a' to append
                output.write(f"{M_c}    {Y_cp}     {omega}    {Tcs}    {Y_f}\n")
plt.scatter(mdm, ycp, color='red', marker='o')

plt.xlabel('Mdm')
plt.ylabel('Y_cp')
plt.title('parameters satisying relic density conditon')
   