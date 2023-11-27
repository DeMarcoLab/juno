import numpy as np

from juno.juno_custom.elements.Herschel.structures import HerschelSettings


def z_a(settings: HerschelSettings):
    return lambda a: a * settings.radius ** settings.exponent

def z_a_p(settings: HerschelSettings):
    return lambda a: settings.exponent * a * settings.radius ** (settings.exponent-1)

def rho_zap(settings: HerschelSettings):
    return lambda a: 1 + (z_a_p(settings)(a) ** 2)

def rho_square(settings: HerschelSettings):
    return lambda a: settings.radius**2 + (settings.z_medium_o-z_a(settings)(a))**2

def rho_square_zap(settings: HerschelSettings):
    return lambda a: settings.radius**2 + (settings.z_medium_o-z_a_p(settings)(a))**2

def rho_numerator(settings: HerschelSettings):
    return lambda a: settings.n_medium_o * ((z_a(settings)(a)-settings.z_medium_o) * z_a_p(settings)(a) + settings.radius)

def rho_numerator_zap(settings: HerschelSettings):
    return lambda a: settings.n_medium_o * ((z_a_p(settings)(a)-settings.z_medium_o) * z_a_p(settings)(a) + settings.radius)

# equation 13
def z_tau(settings: HerschelSettings):
    return lambda a: -settings.thickness + z_a(settings)(a) - settings.z_medium_i

# equation 11
def z_f(settings: HerschelSettings):
    return lambda a: -settings.n_medium_o * settings.z_medium_o + settings.n_lens * settings.thickness + settings.n_medium_i \
        * settings.z_medium_i + np.sign(settings.z_medium_o) * settings.n_medium_o * np.sqrt(rho_square(settings)(a))

def rho_r_term_1(settings: HerschelSettings):
    return lambda a: rho_numerator(settings)(a) / (settings.n_lens * np.sqrt(rho_square_zap(settings)(a)) * rho_zap(settings)(a))

def rho_r_term_2(settings: HerschelSettings):
    return lambda a: z_a_p(settings)(a) / np.sqrt(rho_zap(settings)(a)) * (np.sqrt(1 - (np.square(settings.n_medium_o) * np.square(rho_numerator(settings)(a)) / (np.square(settings.n_lens) * rho_square_zap(settings)(a) * rho_zap(settings)(a)))))    

def rho_r(settings: HerschelSettings):
    return lambda a: rho_r_term_1(settings)(a) - rho_r_term_2(settings)(a)

def rho_z_term_1(settings: HerschelSettings):
    return lambda a: ((settings.n_medium_o * ((z_a(settings)(a)-settings.z_medium_o) * z_a_p(settings)(a) + settings.radius)) * z_a_p(settings)(a)) / (settings.n_lens * np.sqrt(settings.radius**2 + (settings.z_medium_o-z_a(settings)(a))**2) * rho_zap(settings)(a))

def rho_z_term_2(settings: HerschelSettings):
    return lambda a: (np.sqrt(1 - (np.square(settings.n_medium_o * ((z_a(settings)(a)-settings.z_medium_o) * z_a_p(settings)(a) + settings.radius)) /(np.square(settings.n_lens) * (np.square(settings.radius) + np.square(settings.z_medium_o-z_a(settings)(a))) * rho_zap(settings)(a))))) / np.sqrt(rho_zap(settings)(a))

def rho_z(settings: HerschelSettings):
    return lambda a: rho_z_term_1(settings)(a) + rho_z_term_2(settings)(a)

def beta(settings: HerschelSettings):
    return lambda a: z_f(settings)(a) * settings.n_lens + settings.n_medium_i**2 * (settings.radius * rho_r(settings)(a) + z_tau(settings)(a) * rho_z(settings)(a))

def gamma(settings: HerschelSettings):
    return lambda a: settings.n_medium_i**2 * (settings.radius**2 + z_tau(settings)(a)**2) - z_f(settings)(a)**2

# # using + of the beta +/- based off code received in MATLAB
# # TODO: check against actual
def theta(settings: HerschelSettings):
    return lambda a: (-beta(settings)(a) + np.sqrt(-gamma(settings)(a)*(settings.n_medium_i**2 - settings.n_lens**2) + beta(settings)(a)**2))/(settings.n_medium_i**2-settings.n_lens**2)

# def theta(settings: HerschelSettings, parameters: HerschelParameters) -> float:
#     return (-beta(settings, parameters) + np.sqrt(np.square(beta(settings, parameters)) - gamma(settings, parameters))) / gamma(settings, parameters)
#     # theta = lambda a: (-((-n_medium_o * z_medium_o + n_lens * thickness + n_medium_i * z_medium_i + np.sign(z_medium_o) * n_medium_o * np.sqrt(radius**2 + (z_medium_o- z_a(a))**2)) * n_lens + n_medium_i**2 * (radius * (n_medium_o * ((z_a(a)-z_medium_o) * z_a_p(a) + radius) / (n_lens * np.sqrt(radius**2 + (z_medium_o-z_a_p(a))**2) * rho_zap(a)) - (z_a_p(a) / np.sqrt(1 + z_a_p(a)**2)) * (np.sqrt(1 - (np.square(n_medium_o) * np.square(radius + (z_a_p(a) - z_medium_o) * z_a_p(a)) / (np.square(n_lens) * (radius**2 + (z_medium_o-z_a_p(a))**2) *rho_zap(a)))))) +z_tau(a) * (((n_medium_o * ((z_a(a)-z_medium_o) * z_a_p(a) + radius)) * z_a_p(a)) / (n_lens * np.sqrt(radius**2 + (z_medium_o-z_a(a))**2) *rho_zap(a)) + (np.sqrt(1 - (np.square(n_medium_o * ((z_a(a)-z_medium_o) * z_a_p(a) + radius))/(np.square(n_lens) * (np.square(radius) + np.square(z_medium_o-z_a(a))) *rho_zap(a))))) / np.sqrt(rho_zap(a))))) + np.sqrt(((-n_medium_o * z_medium_o + n_lens * thickness + n_medium_i * z_medium_i + np.sign(z_medium_o) * n_medium_o * np.sqrt(radius**2 + (z_medium_o- z_a(a))**2)) * n_lens + n_medium_i**2 * (radius * (n_medium_o * ((z_a(a)-z_medium_o) * z_a_p(a) + radius) / (n_lens * np.sqrt(radius**2 + (z_medium_o-z_a_p(a))**2) * rho_zap(a)) - (z_a_p(a) / np.sqrt(1 + z_a_p(a)**2)) * (np.sqrt(1 - (np.square(n_medium_o) * np.square(radius + (z_a_p(a) - z_medium_o) * z_a_p(a)) / (np.square(n_lens) * (radius**2 + (z_medium_o-z_a_p(a))**2) *rho_zap(a)))))) +z_tau(a) * (((n_medium_o * ((z_a(a)-z_medium_o) * z_a_p(a) + radius)) * z_a_p(a)) / (n_lens * np.sqrt(radius**2 + (z_medium_o-z_a(a))**2) *rho_zap(a)) + (np.sqrt(1 - (np.square(n_medium_o * ((z_a(a)-z_medium_o) * z_a_p(a) + radius))/(np.square(n_lens) * (np.square(radius) + np.square(z_medium_o-z_a(a))) *rho_zap(a))))) / np.sqrt(rho_zap(a)))))**2-(n_medium_i**2 - n_lens**2) * (n_medium_i**2 * (radius**2 +z_tau(a)**2) - (-n_medium_o * z_medium_o + n_lens * thickness + n_medium_i * z_medium_i + np.sign(z_medium_o) * n_medium_o * np.sqrt(radius**2 + (z_medium_o- z_a(a))**2))**2))) / (np.square(n_medium_i) - np.square(n_lens))

def z_b(settings: HerschelSettings):
    return lambda a: z_a(settings)(a) + theta(settings)(a) * rho_z(settings)(a)

def r_b(settings: HerschelSettings):
    return lambda a: settings.radius + theta(settings)(a) * rho_r(settings)(a)

def eq_19_term_1(settings: HerschelSettings):
    return lambda a: (1 - ((z_a(settings)(a) - settings.z_medium_o) / (np.sqrt(np.square(settings.radius) + np.square(-settings.z_medium_o + z_a(settings)(a))))))

def eq_19_term_2(settings: HerschelSettings):
    return lambda a: settings.magnification * settings.n_medium_i / settings.n_medium_o * (1 - (-z_b(settings)(a) + settings.z_medium_i + settings.thickness) / (np.sqrt(np.square(r_b(settings)(a)) + np.square(z_b(settings)(a) - settings.z_medium_i - settings.thickness))))

def eq_19(settings: HerschelSettings):
    return lambda a: eq_19_term_1(settings)(a) - eq_19_term_2(settings)(a)

def eq_19_prime(settings: HerschelSettings):
    return lambda a: - (np.square(settings.radius)) / ((np.square(settings.radius) + np.square(z_a(settings)(a) - settings.z_medium_o)) ** (3/2))
