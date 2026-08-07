import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, quad
from scipy.special import airy, airye, iv

# ==============================================================================
# CONSTANTES PHYSIQUES ET PARAMÈTRES DE BASE
# ==============================================================================
Q_ELEM = 1.602e-19  # Charge élémentaire (C)
EV_TO_J = Q_ELEM  # Conversion eV en Joules
EPSILON_0 = 8.854e-12  # Permittivité du vide (F/m)
EPSILON_R = 3.9  # Permittivité relative du SiO2
EPSILON_SIO2 = EPSILON_0 * EPSILON_R

M_ELEC = 0.42 * 9.10938356e-31  # Masse effective de l'électron (kg)
HBAR = 1.054571817e-34  # Constante de Planck réduite (J.s)
K_B = 1.380649e-23  # Constante de Boltzmann (J/K)
V_F = 1e6  # Vitesse de Fermi (m/s)

WORKFUNCTION_GRAPHENE_EV = 4.5
WORKFUNCTION_SIO2_EV = 3.1
Z_G = 1e-10
D_INT = 3e-10  # Distance graphène-interface (m)
D_TOTAL = 90e-9  # Épaisseur totale (m)
VG_DEFAULT = 2.0  # (V)
VDS_DEFAULT = 0.0  # (V)

E_EV = 0.1  # Énergie (eV)
E_JOULES = E_EV * EV_TO_J
T_SI = D_TOTAL - D_INT
V_TRAP = EV_TO_J * 0.25
X_TRAP = 2e-9
L_BOX = 1e-6
HBAR_OMEGA = 0.02 * EV_TO_J  # Énergie des phonons


# ==============================================================================
# POTENTIEL ET FONCTIONS D'ONDE DE BASE
# ==============================================================================
def compute_potential_profile(
    z,
    z_g,
    d,
    D,
    Vg,
    workfunction_graphene_eV,
    workfunction_SiO2_eV,
    Vds,
):
    """Calcule les profils de potentiel V(z) et U(z)."""
    V = np.zeros_like(z)
    U = np.zeros_like(z)
    t_si = D - d
    denom = EPSILON_0 * t_si + EPSILON_SIO2 * d

    for i, zi in enumerate(z):
        if zi < -z_g:
            V[i] = 0.0
            U[i] = workfunction_graphene_eV - V[i]
        elif zi < 0:
            V[i] = 0.0
            U[i] = -V[i]
        elif zi < d:
            V[i] = (
                EPSILON_SIO2 * (((-Vds) / 2.0) + Vg) / denom
            ) * zi + Vds / 2.0
            U[i] = workfunction_graphene_eV - V[i]
        elif zi <= D:
            V[i] = EPSILON_0 * (((-Vds / 2.0) + Vg) / denom) * (zi - D) + Vg
            U[i] = workfunction_SiO2_eV - V[i]

    return V, U


def show_graph(x, y, xlabel="x", ylabel="y"):
    """Affiche un graphique simple 2D."""
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def integral_of_product_airy(a_coef, b_coef, x0, x1):
    """Calcule les intégrales du produit de fonctions d'Airy."""
    scale = abs(a_coef) ** (2 / 3)

    integral_Ai_Ai, _ = quad(
        lambda x: airy((a_coef * x + b_coef) / scale)[0] ** 2, x0, x1
    )
    integral_Bi_Ai, _ = quad(
        lambda x: airy((a_coef * x + b_coef) / scale)[2]
        * airy((a_coef * x + b_coef) / scale)[0],
        x0,
        x1,
    )
    integral_Bi_Bi, _ = quad(
        lambda x: airy((a_coef * x + b_coef) / scale)[2] ** 2, x0, x1
    )

    return integral_Ai_Ai, integral_Bi_Bi, integral_Bi_Ai


def phi_graphene(x, Vg):
    """Résolution de Schrödinger avec fonctions d'Airy."""
    denom = EPSILON_0 * T_SI + EPSILON_SIO2 * D_INT
    slope2_Vpm = -EPSILON_0 * (((-VDS_DEFAULT) / 2.0) + Vg) / denom
    constant_a_eV = WORKFUNCTION_SIO2_EV - (Vg + slope2_Vpm * D_TOTAL)
    constant_a = constant_a_eV * EV_TO_J
    slope2_Jpm = slope2_Vpm * Q_ELEM

    m = 2 * M_ELEC * slope2_Jpm / (HBAR**2)
    c = -2 * M_ELEC * (E_JOULES - constant_a) / (HBAR**2)
    scale = abs(m) ** (2 / 3)

    z_0 = c / scale
    z_D = (m * D_TOTAL + c) / scale

    Ai_D, Aip_d, Bi_D, Bip_D = airy(z_D + 0j)
    Ai_0, Aip_0, Bi_0, Bip_0 = airy(z_0 + 0j)
    Ais_D, Aips_d, Bis_D, Bips_d = airye(z_D + 0j)
    Ais_0, Aips_0, Bis_0, Bips_0 = airye(z_0 + 0j)

    C = 1
    if Bip_0 > 100 * (Bi_D * Aip_0 / Ai_D):
        B = (
            C
            * np.exp(-abs(2.0 / 3.0 * (z_0 * np.sqrt(z_0 + 0j)).real))
            / Bips_0
        )
    else:
        B = C / (Bip_0 + (Bi_D * Aip_0 / Ai_D))

    A = (
        B
        * np.exp(abs(2.0 / 3.0 * (z_D * np.sqrt(z_D + 0j)).real))
        * np.exp(2.0 / 3.0 * (z_D * np.sqrt(z_D + 0j)).real)
        * Bis_D
        / Ais_D
    )

    z = (m * x + c) / scale
    Ai, Aip, Bi, Bip = airy(z)

    Phi = A * Ai + B * Bi
    Phi_deriv = (A * Aip + B * Bip) * m / scale

    return Phi, Phi_deriv


def phi_trap(x, x_trap, Vg):
    """Fonction d'onde du piège (1D)."""
    K = np.sqrt(-2 * M_ELEC * (E_JOULES - V_TRAP)) / HBAR
    return (
        # 1e-4
        np.sqrt(K / (2 * np.pi))
        * (1 / (x - x_trap))
        * np.exp(-K * np.abs(x - x_trap))
    )


# ==============================================================================
# ÉLÉMENT DE MATRICE ET SEMAINS THERMIQUES
# ==============================================================================
def matrix_element(z, Vg, s, angle, k, V_trap_val):
    """Calcule le carré du module de l'élément de matrice <i|H|f>."""
    kx = k * np.cos(angle)
    ky = k * np.sin(angle)
    K = np.sqrt(2 * M_ELEC * (E_JOULES - V_trap_val) + 0 * 1j) / HBAR
    z_constant = z

    def theta_func(x, y, z_val):
        return (
            (1 / (L_BOX * np.sqrt(2)))
            * (1 + s * np.exp(1j * angle))
            * (np.exp(1j * (kx * x + ky * y)))
        )

    def phi_trap_3d(x, y, z_val):
        r = np.sqrt(x**2 + y**2 + z_val**2)
        return np.sqrt(K / (2 * np.pi)) * (1 / r) * np.exp(-K * r)

    rho_max = 100.0 / (K + 1e-10)

    def g1_real(rho, theta):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        r = np.sqrt(rho**2 + z_constant**2)
        theta_val = theta_func(x, y, z_constant)
        phi_val = phi_trap_3d(x, y, z_constant)
        integrand = (
            np.conj(theta_val).real
            * (z_constant * (K + (1 / r)) / r)
            * phi_val
        )
        return integrand * rho

    def g1_imag(rho, theta):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        r = np.sqrt(rho**2 + z_constant**2)
        theta_val = theta_func(x, y, z_constant)
        phi_val = phi_trap_3d(x, y, z_constant)
        integrand = (
            np.conj(theta_val).imag
            * (z_constant * (K + (1 / r)) / r)
            * phi_val
        )
        return integrand * rho

    def g2_real(rho, theta):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        theta_val = theta_func(x, y, z_constant)
        phi_val = phi_trap_3d(x, y, z_constant)
        return np.conj(theta_val).real * phi_val * rho

    def g2_imag(rho, theta):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        theta_val = theta_func(x, y, z_constant)
        phi_val = phi_trap_3d(x, y, z_constant)
        return np.conj(theta_val).imag * phi_val * rho

    opts = {"epsabs": 1e-16, "epsrel": 1e-5}
    Integral_1_real, _ = dblquad(
        g1_real, 0, 2 * np.pi, lambda t: 0, lambda t: rho_max, **opts
    )
    Integral_1_imag, _ = dblquad(
        g1_imag, 0, 2 * np.pi, lambda t: 0, lambda t: rho_max, **opts
    )
    Integral_2_real, _ = dblquad(
        g2_real, 0, 2 * np.pi, lambda t: 0, lambda t: rho_max, **opts
    )
    Integral_2_imag, _ = dblquad(
        g2_imag, 0, 2 * np.pi, lambda t: 0, lambda t: rho_max, **opts
    )

    phi_val, phi_deriv_val = phi_graphene(z_constant, Vg)

    term1 = phi_val * (Integral_1_real + 1j * Integral_1_imag)
    term2 = (Integral_2_real + 1j * Integral_2_imag) * phi_deriv_val

    result = ((HBAR**2) / (2 * M_ELEC)) * (term1 + term2)
    return np.conj(result) * result


# ==============================================================================
# DISTRIBUTION FERMI-DIRAC & TAUX DE TRANSITION MULTIPHONONS
# ==============================================================================
def compute_WmC(m, S, hbar_omega, T):
    """Probabilité d'échange de multiphonons W_m^C (Formule de Keil)."""
    x = hbar_omega / (2 * K_B * T)
    exp_factor = np.exp(m * x - S / np.tanh(x))
    bessel_factor = iv(m, S / np.sinh(x))
    return exp_factor * bessel_factor


def fermi_dirac(E, T):
    """Distribution de Fermi-Dirac."""
    return 1 / (np.exp(E / (K_B * T)) + 1)


def E_T(m, E_f):
    """Énergie modifiée par les multiphonons."""
    return m * HBAR_OMEGA + E_f


def dos_traps(E_f, m, T):
    """Densité d'états des pièges (Gaussienne)."""
    mu = 0.1
    sigma = 0.1
    energy = E_T(m, E_f)
    return (1e14 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((energy - mu) / sigma) ** 2
    )


def transition_rate(z, Vg, s, angle, k, Ef, M, T=300):
    """Calcul du taux de transition total."""
    print("Ef", Ef)
    gs, gv = 2, 2
    DOS_Ef = gs * gv * np.abs(Ef) / (2 * np.pi * (HBAR * V_F) ** 2)

    m_array = np.arange(-M, M + 1)
    rate_array = np.zeros_like(m_array, dtype=np.float64)

    for m in m_array:
        print("m", m)
        e_t_val = E_T(m, Ef)
        Mif = matrix_element(z, Vg, s, angle, k, e_t_val)
        print("E_T", e_t_val)

        FDt = 1 - fermi_dirac(e_t_val, T)
        FDg = fermi_dirac(Ef, T)

        rate_array[m + M] = (
            (2 * np.pi / HBAR)
            * Mif
            * dos_traps(Ef, m, T)
            * compute_WmC(m, 7, HBAR_OMEGA, T)
            * FDg
            * FDt
            * DOS_Ef
        )
        print("rate_array[m + M]", rate_array[m + M])

    rate = np.sum(rate_array)
    print("Mif", Mif)
    return rate


# ==============================================================================
# SCRIPT PRINCIPAL & TRACÉS
# ==============================================================================
if __name__ == "__main__":
    # --- 1. Profil du Potentiel U(z) ---
    z_eval = np.linspace(-D_TOTAL, D_TOTAL, 10000)
    _, U_values = compute_potential_profile(
        z_eval,
        Z_G,
        D_INT,
        D_TOTAL,
        VG_DEFAULT,
        WORKFUNCTION_GRAPHENE_EV,
        WORKFUNCTION_SIO2_EV,
        VDS_DEFAULT,
    )
    show_graph(z_eval, U_values, xlabel="z (m)", ylabel="U (eV)")

    # --- 2. Fonctions d'onde pour différentes tensions Vg ---
    Vgs = np.arange(2, 22, 4)
    x_eval = np.linspace(0, D_TOTAL, 2000)

    plt.figure()
    for Vg in Vgs:
        phi_val, _ = phi_graphene(x_eval, Vg)
        plt.plot(x_eval, np.abs(phi_val) ** 2, label=f"Vg = {Vg}")

    plt.plot(
        x_eval,
        phi_trap(x_eval, X_TRAP, VG_DEFAULT) ** 2,
        label=f"Vg = {Vgs[-1]} (Trap)",
    )
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- 3. Calcul du taux de transition en fonction de T ---
    z_s = X_TRAP / 2.0
    k_F = E_JOULES / (HBAR * V_F)
    C_ox = EPSILON_SIO2 / D_TOTAL
    Vg_calc = Q_ELEM * (k_F**2) / (np.pi * C_ox)

    print(
        "Transition rate:",
        transition_rate(z_s, Vg_calc, 1, 0, k_F, E_JOULES, 15, 300),
    )

    temperatures = [1, 25, 50, 75, 100]
    plt.figure(figsize=(9, 5))

    for Temp in temperatures:
        rate = transition_rate(z_s, Vg_calc, 1, 0, k_F, E_JOULES, 15, Temp)
        print(f"Transition rate at T={Temp} K: {rate}")
        plt.plot(Temp, rate, "o-", label=f"T = {Temp} K")

    plt.xlabel("Température T (K)", fontsize=12)
    plt.ylabel("Taux de transition", fontsize=12)
    plt.title("Taux de transition en fonction de la température", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()