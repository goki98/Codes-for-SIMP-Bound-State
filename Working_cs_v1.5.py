import numpy as np

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

    # Calculate sigma^t
    sigma_t = 2 * np.pi * sigma_0 / (1 + r)

    # Convert sigma^t to sigma^t / m_x in cm^2/g
    sigma_t_over_mchi = sigma_t / m_x_grams

    return sigma_t_over_mchi

# Given parameters
alpha_x = 3e-4
m_x_GeV = 5000  # MeV
m_phi_MeV = 5  # MeV

# Relative velocities (in km/s)
v_rel_values = np.array([10, 100, 1000, 10000,100000])

# Calculate sigma^t / m_x for each velocity
sigma_t_over_mchi_values = calculate_sigma_t_over_mchi(v_rel_values, alpha_x, m_x_GeV, m_phi_MeV)

# Display results
for v_rel, sigma in zip(v_rel_values, sigma_t_over_mchi_values):
    print(f"v_rel = {v_rel} km/s, sigma^(t) / m_x = {sigma:.3e} cm^2/g")
