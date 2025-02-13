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
for M_c in np.linspace(1*M, 1000*M,100):
    for Y_cp in np.arange(0.001, 1, 0.001):
        L_ch = 0.1
        L_c = 1
        v_h = 246*G
        L_cp = 0.0001
        v_p = 60*M_c
        mu_cp = 0.1*M_c
        c = 0.1
        s = 0.99
        m_1 = 25
        mu_c = 0.5*M_c
        m_2 = 125*G
        g_dm=1
        gs=10.75
        MPl = 1.22e22  # Planck mass in MeV
        #3 to 2 thermal avg cross section
        M2=((-(L_ch * v_h * c - (L_cp * v_p + mu_cp) * s))*(-s*Y_cp))/(4*M_c**2-m_1**2)
        Tcs=(np.sqrt(5)*2*M2**2)/(192*np.pi*M_c**3)
        

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
        const1=2.752*10**5
        const2=(M_c)/(1*M)
        omega=const1*const2*Y_f
        print(M_c,omega,Tcs,Y_f)
        if 0.115<=omega<=0.125:
            rd.append(omega)
            mdm.append(M_c)
            ycp.append(Y_cp)
            with open('RelicDensity_parametr_space_1.txt', 'a') as output:  # Use 'a' to append
                output.write(f"{M_c}    {Y_cp}     {omega}    {Tcs}    {Y_f}\n")
plt.loglog(mdm, ycp, "r--")
plt.xlabel('Mdm')
plt.ylabel('Y_cp')
plt.title('parameters satisying relic density conditon')
   