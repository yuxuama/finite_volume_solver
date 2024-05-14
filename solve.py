import numpy as np
from numba import njit
import h5py
from utils import *

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
    
    N = params[p_N] # Pour la clareté

    # tableaux pour sauvegarder l'état
    
    f = h5py.File(params[p_in], 'r')

    U_old = f['data'][:]
    U = np.zeros_like(U_old)

    # Discretisation de l'espace
    dx = 1 / N

    mask = np.arange(N+2) # Permet de condenser les écritures plus tard

    t = 0

    while t < params[p_T_end]:
        
        max_speed_info = np.max(np.abs(get_speed(U_old, mask)) + get_sound_speed(U_old, mask, params))
        dt = params[p_CFL] * dx / max_speed_info
        
        if t+dt > params[p_T_end]:
            dt = params[p_T_end] - t

        for i in range(1, N+1):
            U[i] = U_old[i] - (dt/dx) * (compute_flux(U_old, i+1, params) - compute_flux(U_old, i, params))
        
        if params[p_BC] == 'neumann':
            U[0] = U[1] 
            U[N+1] = U[N]
        elif params[p_BC] == 'periodic':
            U[0] = U[N]
            U[N + 1] = U[1]

        U_old = U.copy()
        t = t+dt

    with h5py.File(params[p_out], "w") as f:
        dset = f.create_dataset('data', (N+2, 3), dtype=float)
        dset[:] = U
        create_all_attribute(dset, params)
        
    
    return

