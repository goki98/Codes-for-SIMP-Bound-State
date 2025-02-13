import numpy as np
import matplotlib.pyplot as plt

def calculate_sigma_t_over_mchi(v_rel_km_s, alpha_x, m_x_MeV, m_phi_MeV):

    # Convert inputs to CGS units
    hbar_c = 6.582e-22 * 2.998e10  # [6.582e-22 MeV s] * [2.998e10 cm/s]
    MeV_to_g = 1.78266192e-27  # 1 MeV/c^2 = 1.78e-27 g

    # Convert velocity to dimensionless beta_rel
    beta_rel = v_rel_km_s / 2.998e5  # c = 3e5 km/s

    # Convert masses to grams
    m_x_grams = m_x_MeV * MeV_to_g

    # Convert masses to inv_cm
    m_x_inv_cm = m_x_MeV / hbar_c
    m_phi_inv_cm = m_phi_MeV / hbar_c

    # Calculate r
    r = (beta_rel * m_x_MeV / m_phi_MeV)**2

    # Calculate sigma_0 (cross section scale)
    sigma_0 = (alpha_x**2 * m_x_inv_cm**2) / (m_phi_inv_cm**4)
    #Normal cross section
    # Calculate sigma^t
    sigma_t = 2 * np.pi * sigma_0 / (1 + r)
    su=sigma_t
    stu = -4 * np.pi * sigma_0 * (np.log(1 + r) / (r * (2 + r)))
    Tot_N_s=sigma_t+su+stu
    #transfer cross section
    T_st=((4 * np.pi * sigma_0 )/r)*(np.log(1+r)/r-1/(1+r))
    T_su=((4 * np.pi * sigma_0 )/r)*(1-np.log(1+r)/r)
    T_stu=-((4 * np.pi * sigma_0 )/r)*np.log(1+r)/(2+r)
    T_tot=T_st+T_stu+T_su
    #viscosity cross section
    V_st=((8 * np.pi * sigma_0 )/r**2)*(-2+(2+r)*(np.log(1+r)/r))
    V_su=V_st
    V_stu=((8 * np.pi * sigma_0 )/r**2)*(-1+(2*(1+r)*np.log(1+r))/(r*(2+r)))
    V_tot=V_st+V_su+V_stu
    # Convert sigma to sigma / m_x in cm^2/g
    sigma_t_over_mchi = sigma_t / m_x_grams
    Tot_N_s_g=Tot_N_s/m_x_grams
    T_st_g=T_st/m_x_grams
    T_tot_g=T_tot/m_x_grams
    V_st_g=V_st/m_x_grams
    V_tot_g=V_tot/m_x_grams

    return sigma_t_over_mchi,Tot_N_s_g,T_st_g,T_tot_g,V_st_g,V_tot_g

# Given parameters
alpha_x = 3e-4
m_x_GeV = 5000  # MeV
m_phi_MeV = 5  # MeV

# Relative velocities (in km/s)

v_rel_values = np.array([10, 100, 1000, 10000, 100000])



# Calculate sigma^t / m_x for each velocity

st,stot,tt,ttot,vt,vtot = calculate_sigma_t_over_mchi(v_rel_values, alpha_x, m_x_GeV, m_phi_MeV)

# Print results in a formatted way
print(f"{'v_rel (km/s)':>15} {'st':>10} {'stot':>10} {'tt':>10} {'ttot':>10} {'vt':>10} {'vtot':>10}")
for v_rel, s, s_total, t, t_total, v, v_total in zip(v_rel_values, st, stot, tt, ttot, vt, vtot):
    print(f"{v_rel:>15} {s:>10.3e} {s_total:>10.3e} {t:>10.3e} {t_total:>10.3e} {v:>10.3e} {v_total:>10.3e}")
# First plot: st and stot
plt.figure(figsize=(7, 5))
plt.plot(v_rel_values, st, 'o-', label=r'$\sigma^{(t)}/m_\chi$ (st)', color='blue')
plt.plot(v_rel_values, stot, 's-', label=r'$\sigma^{(t)}/m_\chi$ (stot)', color='orange')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$v_\mathrm{rel}$ (km/s)')
plt.ylabel(r'$\sigma/m_\chi$ (cm$^2$/g)')
plt.title(r'$st$ and $stot$ vs $v_\mathrm{rel}$')
plt.legend()
plt.grid(True)
plt.show()

# Second plot: tt and ttot
plt.figure(figsize=(7, 5))
plt.plot(v_rel_values, tt, 'o-', label=r'$\sigma^{(t)}/m_\chi$ (tt)', color='green')
plt.plot(v_rel_values, ttot, 's-', label=r'$\sigma^{(t)}/m_\chi$ (ttot)', color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$v_\mathrm{rel}$ (km/s)')
plt.ylabel(r'$\sigma/m_\chi$ (cm$^2$/g)')
plt.title(r'$tt$ and $ttot$ vs $v_\mathrm{rel}$')
plt.legend()
plt.grid(True)
plt.show()

# Third plot: vt and vtot
plt.figure(figsize=(7, 5))
plt.plot(v_rel_values, vt, 'o-', label=r'$\sigma^{(t)}/m_\chi$ (vt)', color='purple')
plt.plot(v_rel_values, vtot, 's-', label=r'$\sigma^{(t)}/m_\chi$ (vtot)', color='brown')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$v_\mathrm{rel}$ (km/s)')
plt.ylabel(r'$\sigma/m_\chi$ (cm$^2$/g)')
plt.title(r'$vt$ and $vtot$ vs $v_\mathrm{rel}$')
plt.legend()
plt.grid(True)
plt.show()
