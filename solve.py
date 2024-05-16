import numpy as np
from numba import njit, jit, prange
import h5py
from utils import *
import tqdm

# Méthode des volumes finis

@njit
def compute_flux(U, i, j, params, axis, side):
    """Renvoie le flux à l'interface entre les case
    Si axis = 0: renvoie le flux à l'interface gauche de la case (i,j)
    Si axis = 1: renvoir le flux à l'interface basse de la case (i,j)
    `side` permet de définir correctement l'influence du terme source
    On utilise le solveur de Riemann"""
    
    m = 0 # Pour prendre en compte le terme source de masse

    if axis == 0: # Axe des abscisses
        i_prev = i-1
        j_prev = j
        ul = get_speed_x(U[i_prev, j_prev])
        ur = get_speed_x(U[i, j])
    elif axis == 1:
        i_prev = i
        j_prev = j-1
        ul = get_speed_y(U[i_prev, j_prev])
        ur = get_speed_y(U[i, j])

        # Terme source
        dy = params[p_Ly] / params[p_ny]
        m = dy * params[p_g] * 0.5 * (U[i_prev, j_prev, i_mass] + U[i, j, i_mass])


    pl = get_pressure(U[i_prev, j_prev], params)
    pr = get_pressure(U[i, j], params)

    # Paramètre de couplage vitesse pression
    a = 1.1 * max(U[i_prev, j_prev, i_mass] * get_sound_speed(U[i_prev, j_prev], params), U[i, j, i_mass] * get_sound_speed(U[i, j], params))

    # Vitese à l'interface
    u_star = 0.5 * (ul + ur - (pr - pl + m) / a)

    # Pression à l'interface
    p_star = 0.5 * (pl + pr - a * (ur - ul))

    # Grandeur upwind
    if u_star >= 0:
        U_up = U[i_prev, j_prev, :]
    else:
        U_up = U[i, j, :]
    
    # Calcul du flux
    F = U_up * u_star
    F[i_erg] += p_star * u_star
    F[i_erg] += side * m * u_star / 2
    if axis == 0:
        F[i_momx] += p_star
    elif axis == 1:
        F[i_momy] += p_star
        F[i_momy] += side * m / 2
    
    return F

@njit(parallel=True)
def inside_loop(U, U_old, dt, dx, dy, nx, ny, params):
    """Relation de récurrence entre les vecteurs U"""
    for i in prange(1, nx+1):
        for j in prange(1, ny+1):             
            U[i, j] = U_old[i, j] - (dt/dx) * (compute_flux(U_old, i+1, j, params, axis=0, side=1) - compute_flux(U_old, i, j, params, axis=0, side=-1)) - (dt/dy) * (compute_flux(U_old, i, j+1, params, axis=1, side=1) - compute_flux(U_old, i, j, params, axis=1, side=-1))

@njit(parallel=True)
def compute_dt(U, nx, ny, dx, dy, params):
    """Calcule le pas de temps pour la simulation"""
    dt = params[p_T_end] + 2
    for i in prange(1, nx + 1):
        for j in prange(1, ny+1):
            speed_info = np.abs(get_speed(U[i, j])) + get_sound_speed(U[i, j], params)
            dt_loc = (params[p_CFL] / speed_info) * 1. / (1./dx + 1./dy)
            dt = min(dt, dt_loc)
    return dt

def solve(params):
    """Résout le problème du tube de Sod grâce à la méthode des volumes finis sur un temps `T_end`
    en utilisant les conditions initiales données par U_i
    Enregistre les états du système avec une fréquence de `freq_io`"""
    
    nx = params[p_nx]
    ny = params[p_ny]

    # Array des états du système à un instant t et t+dt (shape = (nx, ny, 4))
    
    f = h5py.File(params[p_in], 'r')

    U_old = f['data'][:]
    U = np.ones_like(U_old)

    # Discretisation de l'espace
    dx = params[p_Lx] / nx
    dy = params[p_Ly] / ny

    # Calcul de l'évolution

    t = 0
    pbar = tqdm.tqdm(total=100)
    while t < params[p_T_end]:
        
        dt = compute_dt(U_old, nx, ny, dx, dy, params)
        if t+dt > params[p_T_end]:
            dt = params[p_T_end] - t

        inside_loop(U, U_old, dt, dx, dy, nx, ny, params)
        
        if params[p_BC] == 'neumann':
            neumann(U, nx, ny)
        elif params[p_BC] == 'periodic':
            periodic(U, nx, ny)
        elif params[p_BC] == 'reflex':
            U = reflex(U, params)

        U_old = U.copy()
        t += dt
        pbar.update(100*dt/ params[p_T_end])
    
    pbar.close()

    # Storage in output file
    save(U, params)      
    
    return

