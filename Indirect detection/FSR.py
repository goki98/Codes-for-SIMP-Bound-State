import numpy as np
from scipy.integrate import quad
from labellines import labelLines
import matplotlib.pyplot as plt
from scipy.special import loggamma
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

    
# Parameters in MeV units
M_DM = 150
mx=M_DM     # Dark matter mass [MeV]
mpiC = 139.57     # Mass of charged pion [MeV]
final_states = {
    "e⁺e⁻": 0.511,   # Electron mass [MeV]
    "μ⁺μ⁻": 105.66,  # Muon mass [MeV]
}

Q = 2 * M_DM       # Center-of-mass energy [MeV]

# Energy range setup
Emin = 1e-3  # Minimum photon energy [MeV]
Emax = Q/2  # Maximum photon energy [MeV]
Egamma = np.logspace(np.log10(Emin), np.log10(Emax), 500)  # [MeV]

 # Anhillation and decay calculation
# Matrix elements

M_ele=0.511
M_mu=105.658
M_Pc=139.57 # mass of charged pions
M_p0=135 # mass of neutral pion
# Thermal aveg cross section

#Parametrs for diff flux
r_sun= 8.33
rho_dm=0.3*1e3 # local density of DM is 0.3 GeV expressed in Mev
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




plt.figure(figsize=(8,8))

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

plt.figure(figsize=(8,8))
plt.plot(Egamma, Egamma**2 * Diff_flux_e_C, 'r--', label=rf"$e^\pm$")
plt.plot(Egamma, Egamma**2 * Diff_flux_m_C, 'b--', label=rf"$\mu^\pm$")
plt.plot(Egamma, Egamma**2 * Diff_flux_pc_C, 'g--', label=rf"$\pi^\pm$")

plt.xlabel(r"$E_\gamma$")
plt.ylabel(r"$E^2_\gamma \, d\Phi/d\Omega dE_\gamma$")
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
labelLines(plt.gca().get_lines(), align=True,xvals=[1e0, 1e1, 1e0,1e1], fontsize=16)
plt.grid(which="major", linestyle="--", linewidth=0.8, alpha=0.7)
plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)

plt.text(0.05, 0.93, "(a)", transform=plt.gca().transAxes, fontsize=22)
plt.show()
# plots for decay
# --- COULOMB Plot ---
plt.figure(figsize=(8,8))

# Electrons
if Q > 2 * M_ele:
    spectrum = dN_dEgamma_lep(Egamma, Q, M_ele)
    Diff_flux_e_C1 = prefact_flux_D * ThermCS_Ce1 * spectrum
    Diff_flux_e_C2 = prefact_flux_D * ThermCS_Ce2 * spectrum
    Diff_flux_e_C3 = prefact_flux_D * ThermCS_Ce3* spectrum
    plt.plot(Egamma, Egamma**2 * Diff_flux_e_C1, 'r--', label=rf"$e^\pm n=1$")
    plt.plot(Egamma, Egamma**2 * Diff_flux_e_C2, 'b--', label=rf"$e^\pm n=2$")
    plt.plot(Egamma, Egamma**2 * Diff_flux_e_C3, 'g--', label=rf"$e^\pm n=3$")
else:
    print(f"Skipping e+e-: Q={Q:.1f} MeV < {2*M_ele:.1f} MeV")

# Muons
if Q > 2 * M_mu:
    spectrum = dN_dEgamma_lep(Egamma, Q, M_mu)
    Diff_flux_m_C1 = prefact_flux_D * ThermCS_Cm1 * spectrum
    Diff_flux_m_C2 = prefact_flux_D * ThermCS_Cm2 * spectrum
    Diff_flux_m_C3 = prefact_flux_D * ThermCS_Cm3 * spectrum
    plt.plot(Egamma, Egamma**2 * Diff_flux_m_C1,'r', label=rf"$\mu^\pm n=1$")
    plt.plot(Egamma, Egamma**2 * Diff_flux_m_C2,'b', label=rf"$\mu^\pm n=2$")
    plt.plot(Egamma, Egamma**2 * Diff_flux_m_C3,'g', label=rf"$\mu^\pm n=3$")
else:
    print(f"Skipping μ+μ−: Q={Q:.1f} MeV < {2*M_mu:.1f} MeV")

# Pions
if Q > 2 * mpiC:
    spectrum_pion = dN_dEgamma_pion(Egamma, Q, mpiC)
    Diff_flux_pc_C1 = prefact_flux_D * ThermCS_Cpc1 * spectrum_pion
    Diff_flux_pc_C2 = prefact_flux_D * ThermCS_Cpc2 * spectrum_pion
    Diff_flux_pc_C3 = prefact_flux_D * ThermCS_Cpc3 * spectrum_pion
    plt.plot(Egamma, Egamma**2 * Diff_flux_pc_C1, 'r-.',label=rf"$\pi^\pm n=1$")
    plt.plot(Egamma, Egamma**2 * Diff_flux_pc_C2, 'b-.',label=rf"$\pi^\pm n=2$")
    plt.plot(Egamma, Egamma**2 * Diff_flux_pc_C3, 'g-.',label=rf"$\pi^\pm n=3$")
else:
    print(f"Skipping π+π−: Q={Q:.1f} MeV < {2*mpiC:.1f} MeV")

plt.xlabel(r"$E_\gamma$")
plt.ylabel(r"$E^2_\gamma \, d\Phi/d\Omega dE_\gamma$")
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()
labelLines(plt.gca().get_lines(), align=True,xvals=[1e0, 1e0, 1e1,1e-1,1e-1,1e1,1e0,1e-2,1e0], fontsize=10)
plt.grid(which="major", linestyle="--", linewidth=0.8, alpha=0.7)
plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
plt.text(0.05, 0.93, "(b)", transform=plt.gca().transAxes, fontsize=22)
plt.show()




