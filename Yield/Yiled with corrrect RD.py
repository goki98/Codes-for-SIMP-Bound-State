import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.ticker import LogLocator, NullFormatter
from labellines import labelLines

# Constants
gs = 10.75
g_dm = 1.0
MPl = 1.22e22  # MeV

mx = 100  # MeV

#parametr inthermal cs v2 to achive correct RD
lam_x = 0.8
g_x = 0.7
mx=100
epi=1j*1e-10# +i epsilon factor
# --- Equilibrium yield functions (with safe exp) ---
def y_eq_X(x): 
    return 0.145 * (g_dm / gs) * x**1.5 * np.exp(-x)

def y_eq_B(x): 
    return 0.145 * (g_dm / gs) * 2 * x**1.5 * np.exp(-2*x)

def y_eq_DP(x): 
    return 0.278 * (g_dm / gs)

def entropy(x): 
    return (2 * np.pi**2) / 45 * gs * (mx / x)**3

# --- Coupled Boltzmann equations (rewritten to remove denominators) ---
def dydx_all(x, y, Tcs_tuple):
    yX, yB = y
    (Tcs1, Tcs2, Tcs3, Tcs4, Tcs5, Tcs6,
     Tcs7, Tcs8, Tcs9, Tcs10, Tcs11, Tcs12) = Tcs_tuple
    
    eqX, eqB, eqDP = y_eq_X(x), y_eq_B(x), y_eq_DP(x)
    yDP = eqDP  # relativistic species stays in equilibrium
    s = entropy(x)
    Hm = 1.67 * np.sqrt(gs) * mx**2 / MPl
    #effective cross sections
    cutoff=300
    if x > cutoff:
      tcs1 = 0.0
      tcs2 = 0.0
      tcs4 = 0.0
      tcs5 = 0.0
      tcs6 = 0.0
      tcs9 = 0.0
      tcs10=0
      tcs11 = 0.0
      tcs12 = 0.0
    else:
      tcs1 = Tcs1 / (eqB * eqDP)
      tcs2 = Tcs2 / (eqX**2)
      tcs4 = Tcs4 / (eqB * eqDP)
      tcs5 = Tcs5 / (eqDP**2)
      tcs9 = Tcs9 
      tcs11 = Tcs11 / (eqDP)
      tcs12 = Tcs12 / (eqDP)
      tcs6 = Tcs6 / (eqX**2)
      tcs10 = Tcs10 / (eqX**2)
      
    
    
   
    # --- No denominators with Y_eq ---
    t1 = s**2 * tcs1 * (yX**2 * eqDP * eqB - eqX**2 * yDP * yB)
    t2 = s**2 * tcs2 * (yDP * yB * eqX**2 - eqDP * eqB * yX**2)
    t3 = s**4 * Tcs3 * (yX**4 - eqX**2 * yX**2)
    t4 = s**2 * tcs4 * (yX**4 * eqDP * eqB - eqX**4 * yDP * yB)
    t5 = s**2 * tcs5 * (yX**2 * eqDP**2 - eqX**2 * yDP**2)
    t6 = s**2 * Tcs6 * (yDP**2 * eqX**2 - eqDP**2 * yX**2)
    t9 = s**3 * Tcs9 * (yX**2 * yB - eqB**2 * yX**2)
    t10 = s**2 * tcs10 * (yB**2 * eqX**2 - eqB**2 * yX**2)
    t11 = s**3 * tcs11 * (yX**2 * yB * eqDP - eqX**2 * yDP * yB)
    t12 = s**2 * tcs12 * (yB**2 * eqDP - eqB * yB * yDP)
    
    Prefactor = x / (s * Hm)
    
    dy1 = Prefactor * (t2 - t1 - t3 - t4  + t10+t9 - t11)
    dy2 = Prefactor * (t1 - t2 + t4 - t10 -t9- t12)
    return np.array([dy1, dy2])

# --- Backward Euler solver ---
def backward_euler_coupled(f, x0, y0, x_end, h, Tcs_tuple):
    x_values = [x0]
    y_values = [y0]
    x = x0
    y = np.array(y0)

    while x < x_end:
        def equation(y_new):
            return y_new - y - h * f(x + h, y_new, Tcs_tuple)
        
        y_new = fsolve(equation, y)
        x += h
        x_values.append(x)
        y_values.append(y_new)
        y = y_new

    return np.array(x_values), np.array(y_values)
plt.figure(figsize=(8, 8))
# --- Initial conditions ---
x0 = 1.0
x_end = 1000.0
h = 0.5
y0 = [0.02, 0.01]

# --- Cross-sections ---
Tcs1 = 2.5e-7
Tcs2 = 0.5e-26
Tcs3 = 5.230316e-7
Tcs4 = 4.741107e-06
Tcs5 = 0 
Tcs6 = 0
Tcs7 = 2.5e-10 
Tcs8 = 9.5e-9 
Tcs9 =7.365470e-20
Tcs10 =  1.071169e-03
Tcs11 = 5.030030e-2
Tcs12 = 5.692892e-2

Tcs_tuple = (Tcs1, Tcs2, Tcs3, Tcs4, Tcs5, Tcs6,
             Tcs7, Tcs8, Tcs9, Tcs10, Tcs11, Tcs12)

# --- Solve system ---
x_vals, y_vals = backward_euler_coupled(dydx_all, x0, y0, x_end, h, Tcs_tuple)
y_vals = np.array(y_vals)

# --- Plotting yields ---
labels = [r"$Y_{\chi}$", r"$Y_{B}$"]
colors = ['r', 'g']
for i in range(2):
    plt.loglog(x_vals, y_vals[:, i], color=colors[i], label=labels[i])

# Equilibrium yields
Xf = np.arange(1, 1000, 1.0)
Yield1 = [y_eq_X(x) for x in Xf]
Yield2 = [y_eq_B(x) for x in Xf]
Yield3 = [y_eq_DP(x) for x in Xf]

plt.loglog(Xf, Yield1, "r--")
plt.loglog(Xf, Yield2, "g--")
plt.xlim(5e0,1e3)
plt.ylim(1e-20, 1e0)
#plt.xlim(8e1, 8e2)
plt.xlabel(r'$x = M_\chi / T$')
plt.ylabel('Y')
# --- Compute Relic Density range (0.12 ± 10%) ---
RD_target = 0.12
RD_low = 0.9 * RD_target   # 10% lower
RD_high = 1.1 * RD_target  # 10% higher

# Convert these to yield ranges using your formula
# RD = 2.752e7 * (Y_total)
factor = 2.752e5 * 1e2
Y_total_low = RD_low / factor
Y_total_high = RD_high / factor

# Shade the cosmologically allowed region on the Y-axis
plt.fill_between([1, 1e3], Y_total_low, Y_total_high,
                 color='blue', alpha=0.3,
                 label=r'$\Omega h^2 = 0.12 \pm 10\%$')

plt.grid(True)

# Add ticks on all sides
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
# Label directly along the lines
labelLines(plt.gca().get_lines(), align=True,xvals=[80, 80], fontsize=16)

plt.show()



# --- Final yields ---
print("Final Yields:")
print("Y1 =", y_vals[-1, 0])
print("Y2 =", y_vals[-1, 1])
# --- Compute relic density ---
RD = 2.752e5 *10**2* (y_vals[-1, 0] + y_vals[-1, 1])
print(f"Computed Relic Density Ωh² = {RD:.3e}")

# --- Check if it's approximately 0.12 ---
if np.isclose(RD, 0.12, rtol=0.1):
    print("✅ Relic density is approximately 0.12 (within 10% tolerance).")
else:
    print(f"⚠️ Relic density ({RD:.3e}) differs from 0.12 by {abs(RD - 0.12):.2e}.")