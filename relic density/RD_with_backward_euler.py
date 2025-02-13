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
Tcs = 2.5 * 10**12 *G**-5 # example value for Tcs

# Function for y_eq(x1)
def y_eq(x1):
    return 0.145 * (g_dm / gs) * x1**(3/2) * np.exp(-x1)

# Function for the differential equation
def dydx(x1, y1):
    eq = y_eq(x1)
    return (-0.116 * gs**(3/2) * MPl * Mc**4 * Tcs * (y1**3 - y1**2 * eq)) / (x1**5)

# Backward Euler Method
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


# Solve using backward Euler
x_vals_backward, y_vals_backward = backward_euler(dydx, x0, y0, x_end, h)

# Plot the results

plt.loglog(x_vals_backward, y_vals_backward, "r--", label=r"$Y_{\text{eq}}(x)$ (Yield at vsigma=2.5 * 10**6")

# Plot enhancements
plt.ylim(1e-20, 1e-1)
plt.xlabel('x1')
plt.ylabel('y1')
plt.title('Solution of the differential equation (Backward Euler)')
plt.legend()
plt.grid(True)
plt.show()

# Print the final value of y1
print("Final value of y1:", y_vals_backward[-1])
