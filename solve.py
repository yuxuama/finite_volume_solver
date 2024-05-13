import numpy as np
from numba import njit
import h5py

# Paramètre de la simulation

i_mass = 0 # Pour les tableaux de grandeurs conservatives
i_mom = 1
i_erg = 2

j_mass = 0 # Pour les tableaux de grandeurs primitives
j_press = 1
j_speed = 2

p_gamma = 0 # Pour les paramètres de la simulation
p_N = 1
p_T_end = 2
p_CFL = 3
p_BC = 4
p_freq_out = 5
p_in = 6
p_out = 7

# Utils

@njit
def get_speed(U=np.ndarray, i=int):
    """Renvoie la vitesse du fluide dans la case i"""
    return U[i, i_mom] / U[i, i_mass]

@njit
def get_pressure(U, i, params):
    """Renvoie la pression du fluide dans la case i"""
    erg_kin = 0.5 * U[i, i_mom] ** 2 / U[i, i_mass]
    erg_intern = U[i, i_erg] - erg_kin
    return (params[p_gamma] - 1) * erg_intern

@njit
def get_sound_speed(U, i, params):
    """Renvoie la vitesse du son dans la case de fluide i"""    
    return np.sqrt(params[p_gamma] * get_pressure(U, i, params) / U[i, i_mass])

@njit
def primitive_into_conservative(Q, params):
    """Renvoie le tableau des variables conservatives en partant des variables primitives"""
    U = np.zeros_like(Q)
    U[:, i_mass] = Q[:, j_mass]
    U[:, i_mom] = Q[:, j_mass] * Q[:, j_speed]
    U[:, i_erg] = (Q[:, j_press] / (params[p_gamma] - 1)) + U[:, i_mom]**2 / (2 * U[:, i_mass])

    return U

@njit
def conservative_into_primitive(U, params):
    """Renvoie le tableau des variables primitives en partant des variables conservatives"""
    Q = np.zeros_like(U)
    mask = np.arange(Q.shape[0])
    Q[:, j_mass] = get_speed(U, mask)
    Q[:, j_speed] = get_pressure(U, mask, params)
    
    return Q


# Méthode des volumes finis
# On se place dans les conditions aux limites de Neumann et dans un problème 1D

@njit
def compute_flux(U, i, params):
    """Renvoie le flux à l'interface entre les case i-1 et i (interface gauche)
    On utilise le solveur de Riemann"""
    
    ul = get_speed(U, i-1)
    ur = get_speed(U, i) 
    pl = get_pressure(U, i-1, params)
    pr = get_pressure(U, i, params)

    # Paramètre de couplage vitesse pression
    a = 1.1 * max(U[i-1, i_mass] * get_sound_speed(U, i-1, params), U[i, i_mass] * get_sound_speed(U, i, params))

    # Vitese à l'interface
    u_star = (ul + ur - (pr - pl) / a) / 2

    # Pression à l'interface
    p_star = (pl + pr - a * (ur - ul)) / 2

    # Grandeur upwind
    if u_star >= 0:
        U_up = U[i-1,:].copy()
    else:
        U_up = U[i, :].copy()
    
    # Calcul du flux
    F = U_up * u_star + np.array([0., p_star, p_star * u_star])

    return F

def solve(params):
    """Résout le problème du tube de Sod grâce à la méthode des volumes finis sur un temps `T_end`
    en utilisant les conditions initiales données par U_i
    Enregistre les états du système avec une fréquence de `freq_io`"""
    
    # tableaux pour sauvegarder l'état
    
    f = h5py.File(params[p_in], 'r')

    U_old = f['input'][:]
    U = np.zeros_like(U_old)

    # Discretisation de l'espace
    dx = 1 / params[p_N]

    mask = np.arange(params[p_N]+2) # Permet de condenser les écritures plus tard

    t = 0

    while t < params[p_T_end]:
        
        max_speed_info = np.max(np.abs(get_speed(U_old, mask)) + get_sound_speed(U_old, mask, params))
        cfl = 0.9
        dt = cfl * dx / max_speed_info
        
        if t+dt > params[p_T_end]:
            dt = params[p_T_end] - t

        for i in range(1, params[p_N]+1):
            U[i] = U_old[i] - (dt/dx) * (compute_flux(U_old, i+1, params) - compute_flux(U_old, i, params))
        U[0] = U[1]
        U[params[p_N]+1] = U[params[p_N]]

        U_old = U.copy()
        t = t+dt

    with h5py.File(params[p_out], "w") as f:
        dset = f.create_dataset('output', (params[p_N]+2, 3), dtype=float)
        dset[:] = U
    
    return

