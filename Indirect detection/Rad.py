import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLines


# ======================
# Constants
# ======================
alpha = 1/137.0
m_pi = 139.57   # MeV
m_mu = 105.66   # MeV
m_e  = 0.511    # MeV
f_pi = 130.2    # MeV

# Form factors
FV0 = 0.0254
a = 0.10
FA = 0.0119

# Branching ratios (PDG)
Br_pi_to_mu = 0.999877
Br_pi_to_e  = 1.23e-6
# global plot text size 
plt.rcParams.update({
    'font.size': 18,          # Base font size
    "text.usetex": True,      # Latex switch
    'axes.titlesize': 18,     # Subplot titles
    'axes.labelsize': 21,     # Axis labels
    'xtick.labelsize': 18,    # X-axis ticks
    'ytick.labelsize': 18,    # Y-axis ticks
    'legend.fontsize': 18,    # Legend text
    'figure.titlesize': 18    # Main title
})
# ======================
# Muon spectrum (Eq 4.21)
# ======================
def muon_spectrum(Egamma, Emu=m_mu):
    r = (m_e/m_mu)**2
    x = 2*Egamma/m_mu
    if x <= 0 or x >= 1-r:
        return 0
    term1 = 12*(3 - 2*x*(1-x)**2)*np.log((1-x)/r)
    term2 = x*(1-x)*(46 - 55*x) - 102
    return alpha*(1-x)/(36*np.pi*Egamma) * (term1 + term2)

# ======================
# Pion spectrum pieces
# ======================
def FV(q2): 
    return FV0*(1 + a*q2)

def f_func(x, r):
    term1 = m_pi**2 * x**4 * (FA**2 + FV(x)**2) * (r**2 - r*x + r - 2*(x-1)**2)
    term2 = -12*np.sqrt(2) * f_pi * m_pi * r * (x-1) * x**2 * (FA*(r - 2*x + 1) + x*FV(x))
    term3 = -24 * f_pi**2 * r * (x-1) * (4*r*(x-1) + (x-2)**2)
    return (r + x - 1) * (term1 + term2 + term3)

def g_func(x, r):
    prefactor = 12*np.sqrt(2)*f_pi*r*(x-1)**2*np.log(r/(1-x))
    bracket = (m_pi*x**2*(FA*(x-2*r) - x*FV(x)) +
               np.sqrt(2)*f_pi*(2*r**2 - 2*r*x - x**2 + 2*x - 2))
    return prefactor * bracket


def pion_spectrum(Egamma, m_lepton):
    r = (m_lepton/m_pi)**2
    x = 2*Egamma/m_pi
    if x <= 0 or x >= 1-r:
        return 0
    numerator = f_func(x, r) + g_func(x, r)
    denom = 24*np.pi*m_pi*f_pi**2*(r-1)**2*(x-1)**2*r*x
    return alpha*numerator/denom

# ======================
# Total and individual contributions
# ======================
def spectrum_contributions(Egamma):
    spec_e = Br_pi_to_e * pion_spectrum(Egamma, m_e)
    spec_mu_direct = Br_pi_to_mu * pion_spectrum(Egamma, m_mu)
    Emu = (m_pi**2 + m_mu**2)/(2*m_pi)
    spec_mu_rad = Br_pi_to_mu * muon_spectrum(Egamma, Emu)
    total = spec_e + spec_mu_direct +spec_mu_rad
    return spec_e, spec_mu_direct, spec_mu_rad, total

# ======================
# Plot
# ======================
Egamma_vals = np.linspace(1e-3, m_pi/2, 800)
spec_e, spec_mu_direct, spec_mu_rad, total_pion = zip(*[spectrum_contributions(E) for E in Egamma_vals]) #spec_mu_direct is muon from pion decay,
spec_e         = np.array(spec_e)
spec_mu_direct = np.array(spec_mu_direct)
spec_mu_rad    = np.array(spec_mu_rad)
total_pion     = np.array(total_pion)

#spec_mu_direct is muon from pion decay,
#spec_mu_rad is muon radiative spectrum
#spec_e is electron from pion decay


#Parametrs for diff flux
r_sun= 8.33
d_sun=0.3*1e3 # density of sun is 0.3 GeV expressed in Mev
mx=150
#UNCOMMENT WHICH J FACTOR  RU NEED TO USE
#J_factor=4.698*1e27 #https://iopscience.iop.org/article/10.1088/1475-7516/2020/01/056/pdf
#J_factor=1.30e+29#https://arxiv.org/pdf/1805.08379 this is for galactic center
J_factor=8.32e+24#https://arxiv.org/pdf/1604.05599 this is for DRACO drwaf galacy
#J_factor=1.51e+25#https://arxiv.org/pdf/1604.05599 this is for ursa minor dwarf galaxy
D_factor=6.21e+21 # d factor for draco dwarf
# see https://arxiv.org/pdf/1604.05599 for more J factor value
# Calculating spectrum and ploting the diffrential flux
prefact_flux=0.5*(1/(4*np.pi))*(1/mx)**2 * J_factor
prefact_flux_D=(1/(4*np.pi))*(1/mx) * D_factor
#Thermal Cross section values calcualted using sperate code
ThermCS_Ce=2.3210822072912683e-45

ThermCS_Cm=4.5512011582258656e-37

ThermCS_Cpc=1.9717042130299767e-35

ThermCS_Ce1=3.205831447222757e-53
ThermCS_Ce2=4.007289309028446e-54
ThermCS_Ce3=1.1873449804528728e-54
ThermCS_Cm1=2.0956193849963536e-44
ThermCS_Cm2=2.619524231245443e-45
ThermCS_Cm3=7.761553277764274e-46
ThermCS_Cpc1=9.078793546263072e-42
ThermCS_Cpc2=1.134849193282884e-42
ThermCS_Cpc3=3.362516128245582e-43

#muon radiative flux
Diff_flux_Ann_M_Rad=  prefact_flux * ThermCS_Cm  * spec_mu_rad
Diff_flux_Dec_M_Rad_1= prefact_flux_D * ThermCS_Cm1  * spec_mu_rad
Diff_flux_Dec_M_Rad_2= prefact_flux_D * ThermCS_Cm2  * spec_mu_rad
Diff_flux_Dec_M_Rad_3= prefact_flux_D * ThermCS_Cm3  * spec_mu_rad

# pion radtive flux
# electron
Diff_flux_Ann_E_Rad=  prefact_flux * ThermCS_Cpc  * spec_e
Diff_flux_Dec_E_Rad_1= prefact_flux_D * ThermCS_Cpc1  * spec_e
Diff_flux_Dec_E_Rad_2= prefact_flux_D * ThermCS_Cpc2  * spec_e
Diff_flux_Dec_E_Rad_3= prefact_flux_D * ThermCS_Cpc3  * spec_e
# muon 
Diff_flux_Ann_mp_Rad=  prefact_flux * ThermCS_Cpc * spec_mu_direct
Diff_flux_Dec_mp_Rad_1= prefact_flux_D * ThermCS_Cpc1  * spec_mu_direct
Diff_flux_Dec_mp_Rad_2= prefact_flux_D * ThermCS_Cpc2  * spec_mu_direct
Diff_flux_Dec_mp_Rad_3= prefact_flux_D * ThermCS_Cpc3  * spec_mu_direct
#total pion
Diff_flux_Ann_tot=  prefact_flux * ThermCS_Cpc  * total_pion
Diff_flux_Dec_tot_1= prefact_flux_D * ThermCS_Cpc1  * total_pion
Diff_flux_Dec_tot_2= prefact_flux_D * ThermCS_Cpc2  * total_pion
Diff_flux_Dec_tot_3= prefact_flux_D * ThermCS_Cpc3  * total_pion
#ploting
#muon rad
plt.figure(figsize=(8,8))
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Ann_M_Rad, 'g--', lw=2, label=r"$Ann$")
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Dec_M_Rad_1, 'b-.', lw=2, label=r"$n=1$")
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Dec_M_Rad_2, 'm:', lw=2, label=r"$n=2$")
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Dec_M_Rad_3, 'r-', lw=2, label=r"$n=3$")

plt.xlabel(r"$E_\gamma$")
plt.ylabel(r"$E^2_\gamma \, d\Phi/d\Omega dE_\gamma$")

plt.yscale("log")
plt.xscale("log")
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
labelLines(plt.gca().get_lines(), align=True,xvals=[1e0, 1e0, 1e0,1e1], fontsize=16)
plt.grid(which="major", linestyle="--", linewidth=0.8, alpha=0.7)
plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
plt.text(0.05, 0.93, "(a)", transform=plt.gca().transAxes, fontsize=22)
plt.show()
#pion rad
#electron
plt.figure(figsize=(8,8))
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Ann_E_Rad, 'g--', lw=2, label=r"$Ann$")
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Dec_E_Rad_1, 'b-.', lw=2, label=r"$n=1$")
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Dec_E_Rad_2, 'm:', lw=2, label=r"$n=2$")
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Dec_E_Rad_3, 'r-', lw=2, label=r"$n=3$")
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.xlabel(r"$E_\gamma$")
plt.ylabel(r"$E^2_\gamma \, d\Phi/d\Omega dE_\gamma$")
plt.yscale("log")
plt.xscale("log")
labelLines(plt.gca().get_lines(), align=True,xvals=[1e0, 1e0, 1e0,1e1], fontsize=16)
plt.grid(which="major", linestyle="--", linewidth=0.8, alpha=0.7)
plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
plt.text(0.05, 0.93, "(a)", transform=plt.gca().transAxes, fontsize=22)
#plt.show()
#muon
plt.figure(figsize=(8,8))
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Ann_mp_Rad, 'g--', lw=2, label=r"$Ann$")
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Dec_mp_Rad_1, 'b-.', lw=2, label=r"$n=1$")
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Dec_mp_Rad_2, 'm:', lw=2, label=r"$n=2$")
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Dec_mp_Rad_3, 'r-', lw=2, label=r"$n=3$")

plt.xlabel(r"$E_\gamma $")
plt.ylabel(r"$E^2_\gamma \, d\Phi/d\Omega dE_\gamma$")
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.yscale("log")
plt.xscale("log")
labelLines(plt.gca().get_lines(), align=True,xvals=[1e0, 1e0, 1e0,1e1], fontsize=16)
plt.grid(which="major", linestyle="--", linewidth=0.8, alpha=0.7)
plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
plt.text(0.05, 0.93, "(b)", transform=plt.gca().transAxes, fontsize=22)
#plt.show()
#total
plt.figure(figsize=(8,8))
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Ann_tot, 'g--', lw=2, label=r"$Ann$")
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Dec_tot_1, 'b-.', lw=2, label=r"$n=1$")
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Dec_tot_2, 'm:', lw=2, label=r"$n=2$")
plt.plot(Egamma_vals, Egamma_vals**2*Diff_flux_Dec_tot_3, 'r-', lw=2, label=r"$n=3$")

plt.xlabel(r"$E_\gamma $")
plt.ylabel(r"$E^2_\gamma \, d\Phi/d\Omega dE_\gamma$")
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
plt.yscale("log")
plt.xscale("log")
labelLines(plt.gca().get_lines(), align=True,xvals=[1e0, 1e0, 1e0,1e1], fontsize=16)
plt.grid(which="major", linestyle="--", linewidth=0.8, alpha=0.7)
plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
plt.text(0.05, 0.93, "(b)", transform=plt.gca().transAxes, fontsize=22)
plt.show()


