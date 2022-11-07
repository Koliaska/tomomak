import numpy as np
from scipy import interpolate
from scipy.integrate import simps


"Create constants"
m_p = 1.007276466621        # proton mass, u
m_D = 2.0141017778          # deuterium mass, u
m_e = 5.48579909070*1e-4    # electron mass, u


# Additional functions ###############################################################################################
def conversion_speed_energy(m_u, velocity):
    """ It takes mass (u) and velocity (m/s) of the particle and returns the energy of this particle (eV)"""
    m_kg = m_u * 1.660539 * 1e-27  # kg
    E_eV = (m_kg * velocity**2)/(2 * 1.602176 * 1e-19)
    return E_eV


def conversion_energy_speed(m_u, E_eV):
    """ It takes mass (u) and energy (eV) of the particle and returns the speed of this particle (m/s)"""
    m_kg = m_u * 1.660539 * 1e-27  # kg
    v = np.sqrt(2 * E_eV * 1.602176 * 1e-19 / m_kg)  # m/s
    return v    # m/s


def conversion_eV_J(E_eV):
    """ It takes energy of the particle in eV and returns the energy of this particle in J"""
    E_J = E_eV * 1.602176 * 1e-19    # kg*m^2/s^2
    return E_J


def conversion_J_eV(E_J):
    """ It takes energy of the particle in J and returns the energy of this particle in eV"""
    E_eV = E_J / (1.602176 * 1e-19)    # eV
    return E_eV


def interpolation_values(xnew, x, y):
    """ Interpolate function values on a plane """
    func = interpolate.interp1d(x, y, bounds_error=False, fill_value=(0, 0))
    ynew = func(xnew)      # use interpolation function returned by `interp1d`
    return ynew


# Cross section #######################################################################################################
def sigma_charge_exchange_pH(velocity):
    """ The function returns charge exchange [p+H] cross section value depending on speed (m/s)
    range energy: [1, 10^5] eV

    Args:
        velocity (ndarray): speed matrix (m/s)
    Returns:
        ndarray: cross section matrix   (m^2)
    """
    energy = conversion_speed_energy(m_p, velocity)    # eV
    # sm^ --> m^2
    sigma = 0.6937 * 1e-14 * (1 - 0.155 * np.log10(energy))**2 / (1 + 0.1112 * 1e-14 * np.power(energy, 3.3)) * 1e-4

    for i in range(sigma.size):
        if sigma[i] < 0:
            sigma[i] = 0
    return sigma


def sigma_proton_ionization_pH(velocity):
    """ The function returns proton ionization [p+H] cross section value depending on speed (m/s)
    range energy: [100, 5*10^6] eV

    Args:
        velocity (ndarray): speed matrix (m/s)
    Returns:
        ndarray: cross section matrix   (m^2)
    """
    energy = conversion_speed_energy(m_p, velocity)/1e3     # keV

    arg = 0
    param = [-0.4203309*1e2, 0.3557321*1e1, -0.1045134*1e1, 0.3139238, -0.7454475*1e-1, 0.8459113*1e-2, -0.3495444*1e-3]
    for i in range(len(param)):
        arg += param[i] * (np.log(energy))**i
    sigma = np.exp(arg) * 1e-4                # sm^ --> m^2
    for i in range(sigma.size):
        if sigma[i] < 0:
            sigma[i] = 0
    return sigma


def e_ionization_1s(velocity):
    """ The function returns electron ionization [e+H] cross section value depending on speed (m/s)
        E_min = I = 13.6 eV
    Args:
        velocity (ndarray): electron speed matrix (m/s), it is assumed that the ion is much slower than the electron.
    Returns:
        ndarray: cross section matrix   (m^2)
    """
    energy = conversion_speed_energy(m_e, velocity)    # eV
    I = 13.6       # eV
    A = 0.18450
    B = [-0.032226, -0.034539, 1.4003, -2.8115, 2.2986]

    S = np.zeros([energy.size])
    raz = 1 - I/energy
    matrix = np.transpose([raz] * 5)
    pow_calc = np.power(matrix, [1, 2, 3, 4, 5])
    for i in range(energy.size):
        S[i] = np.sum(B * pow_calc[i])
    sigma = 1e-13/(I * energy) * (A * np.log(energy/I) + S) * 1e-4     # sm^ --> m^2
    for i in range(sigma.size):
        if sigma[i] < 0:
            sigma[i] = 0
    return sigma


# Main functions: rate coefficients for various processes ###########################################################
def isothermal_maxwellian_case(m_a, m_b, T_eV, n_points, sigma, max_v=None):
    """ Nuclear fusion rate coefficients, isothermal Maxwellian case (Ta = Tb = T).

    Args:
        m_a (float): Particle a mass (u)
        m_b (float): Particle b mass (u)
        T_eV (ndarray): Temperature distribution at each point  (1D)  (eV)
        n_points (int): Number of grid points
        sigma (ndarray): Dependence of the scattering cross section[1] (m^2) on speed[0] (m/s)  (2D)
        max_v (float): Upper limit of speed range  (m/s). Optional.

    Returns:
        ndarray: rate coefficients at each point
    """
    if max_v is None:
        max_v = 5 * conversion_energy_speed(min(m_a, m_b), max(T_eV))
    if len(sigma[:, 0]) < len(sigma[0]):
        sigma = np.transpose(sigma)

    v0 = 2.188 * 1e6                                                        # m/s
    T_J = T_eV * 1.602176 * 1e-19                                           # kg*m^2/s^2
    reduced_mass = m_a*m_b/(m_a + m_b) * 1.660539 * 1e-27                   # kg
    velocity = np.arange(0, 2*max_v, 1e3)                                   # m/s
    y = np.power(velocity, 2) / v0**2                                       # dimensionless
    sigma_v = interpolation_values(velocity, sigma[:, 0], sigma[:, 1])      # m^2
    R = np.zeros([n_points])
    for i in range(n_points):
        gamma = reduced_mass * np.power(v0, 2) / (2 * T_J[i])             # dimensionless
        f_y = y * np.exp(-gamma * y)                                      # dimensionless
        integral = simps(sigma_v * f_y, y)
        R[i] = 2/np.sqrt(np.pi) * v0 * np.power(gamma, 3/2) * integral     # m^3/s
    return R


def monoenergetic_beam_maxwell_target(V_alpha, m_beta, T_beta, sigma, v_beta_max=None):
    """For the monoenergetic and monodirectional distribution fα(uα) = δ (uα − V) of species α
     interacting with a Maxwellian target β.

    Args:
        V_alpha (float) : Speed of alpha particle (m/s)
        m_beta (float) : Mass target particle (u)
        T_beta (float) : Target temperature at point (eV)
        sigma (ndarray): Dependence of the scattering cross section[1] (m^2) on speed[0] (m/s)  (2D)
        v_beta_max (float): Max velocity maxwellian target (m/s). Optional.
    Returns:
        float: reaction rate coefficient
    """
    if v_beta_max is None:
        v_beta_max = conversion_energy_speed(m_beta, T_beta)       # m/s
    if len(sigma[:, 0]) < len(sigma[0]):
        sigma = np.transpose(sigma)

    V_rell = np.arange(0, 5 * (v_beta_max + V_alpha), 1e3)             # m/s
    v0 = 2.188 * 1e6                                                   # m/s
    T_beta = T_beta * 1.602176 * 1e-19                                 # kg*m^2/s^2
    m_beta = m_beta * 1.660539 * 1e-27                                 # kg

    A = m_beta * (v0 ** 2) / (2 * T_beta)

    y = np.power(V_rell, 2) / (v0 ** 2)                                       # dimensionless
    B = V_alpha / v0                                                          # dimensionless
    F_y = np.sqrt(y) * (np.exp(2 * A * B * np.sqrt(y) - A * y - A * B ** 2) - np.exp(
        -2 * A * B * np.sqrt(y) - A * y - A * B ** 2))                                # dimensionless
    sigma_new = interpolation_values(V_rell, sigma[:, 0], sigma[:, 1])                # m^2
    integral = simps(sigma_new * F_y, y)
    R = (v0 / (2 * np.sqrt(np.pi))) * (np.sqrt(A) / B) * integral
    return R


def isotropically_beam_maxwell_target(V_alpha, func_alpha, m_beta, T_beta, sigma, n_points):
    """If the velocity distribution of species α is expressed by an isotropic function F(v_α), and the distribution of
     species β is Maxwellian, this function return reaction rate coefficient.

    Args:
        V_alpha (ndarray) : Alpha particle speed  (m/s)
        func_alpha(ndarray): Velocity distribution of species α: F(v_α), [n_points, n_velocity], 1/m3/(m/s)
        m_beta (float) : Particle target mass (u)
        T_beta (ndarray) : Target surface (point) temperature (eV)
        sigma (ndarray): Dependence of the scattering cross section[1] (m^2) on speed[0] (m/s)  (2D)
        n_points (int): Number of grid points
    Returns:
        float: reaction rate coefficient
    """
    if len(sigma[:, 0]) < len(sigma[0]):
        sigma = np.transpose(sigma)

    R = np.zeros([n_points])
    for i in range(n_points):
        print('n_point', i)
        R_inside = np.zeros([V_alpha.size])

        for j in range(V_alpha.size):
            R_inside[j] = monoenergetic_beam_maxwell_target(V_alpha[j], m_beta, T_beta[i], sigma)

        R[i] = simps(func_alpha[i] * R_inside, V_alpha)           # 1/s
    return R
