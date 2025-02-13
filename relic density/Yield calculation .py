import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from hepunits import MeV as M
from hepunits import GeV as G

# Constants 
gs = 10.75  #  value for gs
g_dm = 1.0  # value for g_dm = 2s(s+1) for scalar s=0
MPl = 1.22e22  # Planck mass in MeV
Mc = 100 * M  # example value for Mc
rd = []
Yield = []
Xf = []
# example value for Tcs

# Function for y_eq(x1)
def y_eq(x1):
    return 0.145 * (g_dm / gs) * x1**(3/2) * np.exp(-x1)

# Function for the boltzman equation
def dydx(x1, y1):
    eq = y_eq(x1)
    return (-0.116 * gs**(3/2) * MPl * Mc**4 * Tcs * (y1**3 - y1**2 * eq)) / (x1**5)

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

# Initial conditions
x0 = 1.0  # initial x1 value
y0 = 0.1  # initial y1 value
x_end = 1000.0  # final x1 value
h = 0.1  # step size
tcs=[ 2.5 * 10**4 *G**-5 , 2.5 * 10**6 *G**-5 , 2.5 * 10**8 *G**-5 ]
colors = ['r', 'g', 'b']  # Different colors for each Tcs
labels = [r"$T_{\text{cs}} = 2.5 \times 10^4 \, \text{GeV}$", 
          r"$T_{\text{cs}} = 2.5 \times 10^6 \, \text{GeV}$", 
          r"$T_{\text{cs}} = 2.5 \times 10^8 \, \text{GeV}$"]
# solving yield for arbitary thermal avg cs for freeze out conditon
for Tcs, color, label in zip(tcs, colors, labels):
    # Solve using backward Euler for each Tcs
    x_vals_backward, y_vals_backward = backward_euler(dydx, x0, y0, x_end, h)
    
    # Plot the results for freezeout
    plt.loglog(x_vals_backward, y_vals_backward, color=color, label=label)
# yield for equllibrium condition
for X_f in np.arange(1, 1000, 0.001):  # Avoid X_f = 0 for numerical stability
    # Physical constants and parameters
    gs = 10.75
    g_dm = 1
    g_dm_s = g_dm**2 / np.sqrt(gs)
  
    
    # Equilibrium yield
    Y_eq = 0.145 * (g_dm / gs) * X_f**(3/2) * np.exp(-X_f)
    Xf.append(X_f)
    Yield.append(Y_eq)
    print(X_f, Y_eq)

# plot for equlibrium yield
plt.loglog(Xf, Yield, "r--", label=r"$Y_{\text{eq}}(x)$ (Equilibrium Yield)")
# genral plot details
plt.ylim(1e-20, 1e-2)
plt.xlabel('xf=Mdm/Tf')
plt.ylabel('yield')
plt.title('Yield at thermal equillibrium and freeze-out')
plt.legend()
plt.grid(True)
plt.show()

# Print the final value of y1
print("Final value of y1:", y_vals_backward[-1])
