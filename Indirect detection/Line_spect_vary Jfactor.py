import numpy as np
from scipy.integrate import quad
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from labellines import labelLines
from scipy.special import loggamma
#constants
alpha = 1/137.0
m_pi = 139.57   # MeV
mpiC=m_pi
m_mu = 105.66   # MeV
m_e  = 0.511    # MeV
f_pi = 130.2    # MeV

# Form factors
FV0 = 0.0254
a = 0.10
FA = 0.0119

# Branching ratios (PDG)
Br_pi_to_mu = 0.999877
Br_pi_to_e  = 1.23e-4

M_ele=0.511
M_mu=105.658
M_Pc=139.57 # mass of charged pions
M_p0=135 # mass of neutral pion
M_DM = 150 
mx=M_DM 
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

# Thermal aveg cross section
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

#Parametrs for diff flux
prefact_flux=0.5*(1/(4*np.pi))*(1/mx)**2 
prefact_flux_D=(1/(4*np.pi))*(1/mx)
#UNCOMMENT WHICH J FACTOR  RU NEED TO USE
#J_factor=4.698*1e27 #https://iopscience.iop.org/article/10.1088/1475-7516/2020/01/056/pdf
#J_factor=1.30e+29#https://arxiv.org/pdf/1805.08379 this is for galactic center
#J_factor=8.32e+24#https://arxiv.org/pdf/1604.05599 this is for DRACO drwaf galacy

def dN_dEgamma_lep(Egamma, Q, mf, alpha=1/137.036):
    """
    Compute the FSR spectrum dN/dEgamma for dark matter annihilation in MeV units.
    
    Parameters:
        Egamma (float or array): Photon energy [MeV]
        Q (float): Center-of-mass energy = 2 * M_DM [MeV]
        mf (float): Mass of final-state particle (e, μ, π) [MeV]
        alpha (float): Fine structure constant (default: 1/137.036)
        
    Returns:
        float or array: dN/dEgamma value(s) [MeV^{-1}]
    """
    # Convert scalar input to array for consistent processing
    is_scalar = np.isscalar(Egamma)
    Egamma = np.array([Egamma]) if is_scalar else np.asarray(Egamma)
    
    # Initialize result with zeros
    result = np.zeros_like(Egamma)
    
    # Check kinematic thresholds
    mu = mf / Q
    mu2 = mu**2
    denom = 1 - 4 * mu2
    
    # Return zeros if final state is kinematically forbidden
    if denom <= 0 or Q <= 2 * mf:
        return result[0] if is_scalar else result
    
    # Compute x = 2*Egamma/Q
    x = 2 * Egamma / Q
    
    # Define valid energy range: 0 < Egamma < Q/2 * (1 - 4mu^2)
    valid_mask = (x > 0) & (x < 1 - 4 * mu2)
    x_valid = x[valid_mask]
    Egamma_valid = Egamma[valid_mask]
    
    # Calculate intermediate terms
    term_val = np.sqrt(1 - 4 * mu2 / (1 - x_valid))
    log_part = 2 * np.arctanh(term_val)  # Equivalent to log((1+term_val)/(1-term_val))
    
    term1 = 1 - x_valid - 6 * mu2
    term2 = x_valid + 4 * mu2
    bracket = (2 * term1 + term2**2) * log_part - 2 * denom * (1 - x_valid) * term_val
    
    prefactor = alpha / (Egamma_valid * np.pi * denom**1.5)
    result_valid = prefactor * bracket
    
    # Update result for valid energies
    result[valid_mask] = result_valid
    
    return result[0] if is_scalar else result
def dN_dEgamma_pion(Egamma, Q, mf, alpha=1/137.036):
    """
    Compute the FSR spectrum dN/dEgamma for dark matter annihilation in MeV units.
    
    Parameters:
        Egamma (float or array): Photon energy [MeV]
        Q (float): Center-of-mass energy = 2 * M_DM [MeV]
        mf (float): Mass of final-state particle (e, μ, π) [MeV]
        alpha (float): Fine structure constant (default: 1/137.036)
        
    Returns:
        float or array: dN/dEgamma value(s) [MeV^{-1}]
    """
    # Convert scalar input to array for consistent processing
    is_scalar = np.isscalar(Egamma)
    Egamma = np.array([Egamma]) if is_scalar else np.asarray(Egamma)
    
    # Initialize result with zeros
    result = np.zeros_like(Egamma)
    
    # Check kinematic thresholds
    mu = mf / Q
    mu2 = mu**2
    denom = 1 - 4 * mu2
    
    # Return zeros if final state is kinematically forbidden
    if denom <= 0 or Q <= 2 * mf:
        return result[0] if is_scalar else result
    
    # Compute x = 2*Egamma/Q
    x = 2 * Egamma / Q
    
    # Define valid energy range: 0 < Egamma < Q/2 * (1 - 4mu^2)
    valid_mask = (x > 0) & (x < 1 - 4 * mu2)
    x_valid = x[valid_mask]
    Egamma_valid = Egamma[valid_mask]
    
    # Calculate intermediate terms
    term_val = np.sqrt(1 - 4 * mu2 / (1 - x_valid))
    log_part = 2 * np.arctanh(term_val)  # Equivalent to log((1+term_val)/(1-term_val))
    
    term1 = 1 - x_valid - 2 * mu2
    term2 = 1-x_valid 
    bracket = term1 * log_part -  term2* term_val
    
    prefactor = (2*alpha) / (Egamma_valid * np.pi * np.sqrt(denom))
    result_valid = prefactor * bracket
    
    # Update result for valid energies
    result[valid_mask] = result_valid
    
    return result[0] if is_scalar else result
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
    total = spec_e + spec_mu_direct + spec_mu_rad
    return spec_e, spec_mu_direct, spec_mu_rad, total
# Calculating spectrum and ploting the diffrential flux
 # we calculate bse flux without j factor and contro plot it
# FSR spectrum calc
# Energy range setup

Q = 2 * M_DM       # Center-of-mass energy [MeV]
Emin = 1e-3  # Minimum photon energy [MeV]
Emax = Q/2  # Maximum photon energy [MeV]
Egamma = np.logspace(np.log10(Emin), np.log10(Emax), 500)  # [MeV]
#Anhillation
# Electrons
if Q > 2 * M_ele:
    spectrum = dN_dEgamma_lep(Egamma, Q, M_ele)
    Diff_flux_e_C = prefact_flux * ThermCS_Ce * spectrum
   
    
else:
    print(f"Skipping e+e-: Q={Q:.1f} MeV < {2*M_ele:.1f} MeV")

# Muons
if Q > 2 * M_mu:
    spectrum = dN_dEgamma_lep(Egamma, Q, M_mu)
    Diff_flux_m_C = prefact_flux * ThermCS_Cm * spectrum

else:
    print(f"Skipping μ+μ−: Q={Q:.1f} MeV < {2*M_mu:.1f} MeV")

# Pions
if Q > 2 * mpiC:
    spectrum_pion = dN_dEgamma_pion(Egamma, Q, mpiC)
    Diff_flux_pc_C = prefact_flux *ThermCS_Cpc * spectrum_pion
 
else:
    print(f"Skipping π+π−: Q={Q:.1f} MeV < {2*mpiC:.1f} MeV")
#decay

# Electrons
if Q > 2 * M_ele:
    spectrum = dN_dEgamma_lep(Egamma, Q, M_ele)
    Diff_flux_e_C1 = prefact_flux_D * ThermCS_Ce1 * spectrum
    Diff_flux_e_C2 = prefact_flux_D * ThermCS_Ce2 * spectrum
    Diff_flux_e_C3 = prefact_flux_D * ThermCS_Ce3* spectrum
  
else:
    print(f"Skipping e+e-: Q={Q:.1f} MeV < {2*M_ele:.1f} MeV")

# Muons
if Q > 2 * M_mu:
    spectrum = dN_dEgamma_lep(Egamma, Q, M_mu)
    Diff_flux_m_C1 = prefact_flux_D * ThermCS_Cm1 * spectrum
    Diff_flux_m_C2 = prefact_flux_D * ThermCS_Cm2 * spectrum
    Diff_flux_m_C3 = prefact_flux_D * ThermCS_Cm3 * spectrum
   
else:
    print(f"Skipping μ+μ−: Q={Q:.1f} MeV < {2*M_mu:.1f} MeV")

# Pions
if Q > 2 * mpiC:
    spectrum_pion = dN_dEgamma_pion(Egamma, Q, mpiC)
    Diff_flux_pc_C1 = prefact_flux_D * ThermCS_Cpc1 * spectrum_pion
    Diff_flux_pc_C2 = prefact_flux_D * ThermCS_Cpc2 * spectrum_pion
    Diff_flux_pc_C3 = prefact_flux_D * ThermCS_Cpc3 * spectrum_pion
 
else:
    print(f"Skipping π+π−: Q={Q:.1f} MeV < {2*mpiC:.1f} MeV")
#Radiative
#Egamma_vals = np.linspace(1e-3, m_pi/2, 800)
spec_e, spec_mu_direct, spec_mu_rad, total_pion = zip(*[spectrum_contributions(E) for E in Egamma]) #spec_mu_direct is muon from pion decay,
spec_e         = np.array(spec_e)
spec_mu_direct = np.array(spec_mu_direct)
spec_mu_rad    = np.array(spec_mu_rad)
total_pion     = np.array(total_pion)
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
#total
Diff_flux_Ann_tot=  prefact_flux * ThermCS_Cpc  * total_pion
Diff_flux_Dec_tot_1= prefact_flux_D * ThermCS_Cpc1  * total_pion
Diff_flux_Dec_tot_2= prefact_flux_D * ThermCS_Cpc2  * total_pion
Diff_flux_Dec_tot_3= prefact_flux_D * ThermCS_Cpc3  * total_pion


#electron
if Q > 2 * M_ele:
    Elect_Tot=(Diff_flux_e_C1+Diff_flux_e_C+Diff_flux_e_C2+Diff_flux_e_C3) # electron in Rad is due to pion decay so they add with pion psectrum
    
else:
    print(f"Skipping e+e-: Q={Q:.1f} MeV < {2*M_ele:.1f} MeV")

# Muons
if Q > 2 * M_mu:
    Muon_tot=Diff_flux_m_C+Diff_flux_m_C1+Diff_flux_m_C2+Diff_flux_m_C3+Diff_flux_Ann_M_Rad+Diff_flux_Dec_M_Rad_1+Diff_flux_Dec_M_Rad_2+Diff_flux_Dec_M_Rad_3
    
else:
    print(f"Skipping μ+μ−: Q={Q:.1f} MeV < {2*M_mu:.1f} MeV")

# Pions
if Q > 2 * mpiC:
    pion_tot=Diff_flux_pc_C +Diff_flux_pc_C1+Diff_flux_pc_C2+Diff_flux_pc_C3+Diff_flux_Ann_tot+Diff_flux_Dec_tot_1+Diff_flux_Dec_tot_2+Diff_flux_Dec_tot_3
    
else:
    print(f"Skipping π+π−: Q={Q:.1f} MeV < {2*mpiC:.1f} MeV")

#electron
if Q > 2 * M_ele:
    Elect_Tot_Dec=(Diff_flux_e_C1+Diff_flux_e_C2+Diff_flux_e_C3) # electron in Rad is due to pion decay so they add with pion psectrum
    
else:
    print(f"Skipping e+e-: Q={Q:.1f} MeV < {2*M_ele:.1f} MeV")

# Muons
if Q > 2 * M_mu:
    Muon_tot_dec=Diff_flux_m_C1+Diff_flux_m_C2+Diff_flux_m_C3+Diff_flux_Dec_M_Rad_1+Diff_flux_Dec_M_Rad_2+Diff_flux_Dec_M_Rad_3
    
else:
    print(f"Skipping μ+μ−: Q={Q:.1f} MeV < {2*M_mu:.1f} MeV")

# Pions
if Q > 2 * mpiC:
    pion_tot_dec= Diff_flux_pc_C1+Diff_flux_pc_C2+Diff_flux_pc_C3+Diff_flux_Dec_tot_1+Diff_flux_Dec_tot_2+Diff_flux_Dec_tot_3
    
else:
    print(f"Skipping π+π−: Q={Q:.1f} MeV < {2*mpiC:.1f} MeV")

#electron
if Q > 2 * M_ele:
    Elect_Tot_ann=(Diff_flux_e_C) # electron in Rad is due to pion decay so they add with pion psectrum
    
else:
    print(f"Skipping e+e-: Q={Q:.1f} MeV < {2*M_ele:.1f} MeV")

# Muons
if Q > 2 * M_mu:
    Muon_tot_ann=Diff_flux_m_C+Diff_flux_Ann_M_Rad
    
else:
    print(f"Skipping μ+μ−: Q={Q:.1f} MeV < {2*M_mu:.1f} MeV")

# Pions
if Q > 2 * mpiC:
    pion_tot_ann=Diff_flux_pc_C +Diff_flux_Ann_tot
    
else:
    print(f"Skipping π+π−: Q={Q:.1f} MeV < {2*mpiC:.1f} MeV")
# ===========================

# ===========================
# Differential spectra
# ===========================
Diff_Ann = (
    Diff_flux_e_C
    + Diff_flux_m_C
    + Diff_flux_pc_C
    + Diff_flux_Ann_M_Rad
    + Diff_flux_Ann_tot
)

Diff_Dec = (
    Diff_flux_Dec_M_Rad_1 + Diff_flux_Dec_M_Rad_2 + Diff_flux_Dec_M_Rad_3
    + Diff_flux_Dec_tot_1 + Diff_flux_Dec_tot_2 + Diff_flux_Dec_tot_3
    + Diff_flux_e_C1 + Diff_flux_e_C2 + Diff_flux_e_C3
    + Diff_flux_m_C1 + Diff_flux_m_C2 + Diff_flux_m_C3
    + Diff_flux_pc_C1 + Diff_flux_pc_C2 + Diff_flux_pc_C3
)

Grand_Tot_diff_flux = Diff_Ann + Diff_Dec

# ===========================
# J and D factors
# ===========================
J_factors = {
    "Draco Dwarf":     8.32e26,
    "Galactic Center": 1.30e31,
    "Coma Cluster":    1.99e24
}

D_factors = {
    "Draco Dwarf":     6.32e21,
    "Galactic Center": 7.30e27,
    "Coma Cluster":    3.69e19
}

# ======================================================
# 1) PLOT — ANNIHILATION ONLY WITH LABELLINES
# ======================================================

plt.figure(figsize=(8,8))

for name, Jval in J_factors.items():
    flux_ann = Egamma**2 * Diff_Ann * prefact_flux * Jval
    plt.loglog(Egamma, flux_ann, label=name)

plt.xlabel(r"$E_\gamma$")
plt.ylabel(r"$E_\gamma^2 \, d\Phi/dE_\gamma d\Omega$")
plt.grid(which="both", linestyle="--", alpha=0.6)

# Label lines directly
labelLines(plt.gca().get_lines(), align=True,
           xvals=[1e-1, 3e-1, 1e0])   # choose x-positions
plt.text(0.02, 0.96, "(a)", transform=plt.gca().transAxes, fontsize=22)
plt.show()

# ======================================================
# 2) PLOT — DECAY ONLY WITH LABELLINES
# ======================================================

plt.figure(figsize=(8,8))

for name, Dval in D_factors.items():
    flux_dec = Egamma**2 * Diff_Dec * prefact_flux_D * Dval
    plt.loglog(Egamma, flux_dec, label=name)

plt.xlabel(r"$E_\gamma$")
plt.ylabel(r"$E_\gamma^2 \, d\Phi/dE_\gamma d\Omega$")
plt.grid(which="both", linestyle="--", alpha=0.6)

labelLines(plt.gca().get_lines(), align=True,
           xvals=[1e-1, 3e-1, 1e0])

plt.text(0.02, 0.96, "(b)", transform=plt.gca().transAxes, fontsize=22)
plt.show()

# ======================================================
# 3) PLOT — GRAND TOTAL WITH LABELLINES
# ======================================================

plt.figure(figsize=(8,8))

for name in J_factors.keys():

    Jval = J_factors[name]
    ann_flux = Egamma**2 * Diff_Ann * prefact_flux * Jval

    Dval = D_factors[name]
    dec_flux = Egamma**2 * Diff_Dec * prefact_flux_D * Dval

    tot_flux = ann_flux + dec_flux

    plt.loglog(Egamma, tot_flux, label=name)

plt.xlabel(r"$E_\gamma$")
plt.ylabel(r"$E_\gamma^2 \, d\Phi/dE_\gamma d\Omega$")
plt.grid(which="both", linestyle="--", alpha=0.6)

labelLines(plt.gca().get_lines(), align=True,
           xvals=[1e-1, 3e-1, 1e0])


plt.show()






