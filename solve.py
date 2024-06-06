import numpy as np
from numba import njit, prange
import in_file_maker as ifm
from utils import *
import tqdm

func_dict={
    'sod_shock_tube': ifm.sod_shock_tube,
    'riemann_problem_2d': ifm.riemann_problem_2d,
    'rt_instability': ifm.rt_instability,
    'hydrostatic': ifm.hydrostatic,
    'simple_convection': ifm.simple_convection,
    'simple_diffusion': ifm.simple_diffusion,
    'layer': ifm.layer,
    'diffusive_layer': ifm.diffusive_layer,
    'resume simulation': ifm.resume_simulation
}

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
    cl = get_sound_speed(U[i_prev, j_prev], params)
    cr = get_sound_speed(U[i, j], params)

    # Paramètre de couplage vitesse pression
    a = 1.1 * max(U[i_prev, j_prev, i_mass] * cl, U[i, j, i_mass] * cr)

    # Vitese à l'interface
    u_star = 0.5 * (ul + ur - (pr - pl + m) / a)

    # Pression à l'interface
    Ma = np.abs(u_star / min(cr, cl))
    l_Ma_corr = min(Ma, 1) # Low Mach correction
    p_star = 0.5 * (pl + pr - a * l_Ma_corr * (ur - ul))

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

@njit
def t(U, i, j, ny, T0, params):
    """Calcul la température en prenant en compte la température des bords"""
    if j==0 or j==ny+1:
        return T0[i, j]
    return get_temp(U[i, j], params)

@njit(parallel=True)
def inside_loop(U, U_old, T0, dt, dx, dy, nx, ny, params):
    """Applique la relation de récurrence au vecteur U"""
    for i in prange(1, nx+1):
        for j in prange(1, ny+1):
            # On applique le transport             
            U[i, j] = U_old[i, j] - (dt/dx) * (compute_flux(U_old, i+1, j, params, axis=0, side=1) - compute_flux(U_old, i, j, params, axis=0, side=-1)) - (dt/dy) * (compute_flux(U_old, i, j+1, params, axis=1, side=1) - compute_flux(U_old, i, j, params, axis=1, side=-1))

            # On applique les termes sources de chaleur
            # Diffusion thermique
            if params[p_k] != 0: # En soi pas important mais pour le temps de calcul
                Toldold = get_temp(U_old[i, j], params)
                Tdiff = (get_temp(U_old[i+1, j], params) +  get_temp(U_old[i-1, j], params) - 2 * Toldold) / (dx*dx)
                Tdiff += (t(U_old, i, j+1, ny, T0, params) + t(U_old, i, j-1, ny, T0, params) - 2 * Toldold) / (dy*dy)
                Tdiff *= 0.5 * params[p_k] * dt
                U[i, j, i_erg] = U[i, j, i_erg] + U[i, j, i_mass] * params[p_cv] * Tdiff
            
            # Rappel thermique (buoyancy)
            a = params[p_gamma] * params[p_ht] * dt
            Told = get_temp(U[i, j], params)
            Tnew = (Told - a * T0[i, j]) / (1 - a)
            U[i, j, i_erg] = U[i, j, i_erg] + U[i, j, i_mass] * params[p_cv] * (Tnew - Told)


@njit(parallel=True)
def compute_dt(U, nx, ny, dx, dy, params):
    """Calcule le pas de temps pour la simulation"""
    dt = params[p_T_end] + 1 # Il ne pourra pas être plus grand car on le clamp à Tend (dans la boucle principale)
    if params[p_k] != 0:
        dt_loc_diff = (params[p_CFL] / params[p_k]) * 1. / (1./(dx**2) + 1./(dy**2))
    else:
        dt_loc_diff = params[p_T_end] + 1
    for i in prange(1, nx + 1):
        for j in prange(1, ny+1):
            speed_info = np.abs(get_speed(U[i, j])) + get_sound_speed(U[i, j], params)
            dt_loc_ad = (params[p_CFL] / speed_info) * 1. / (1./dx + 1./dy)
            dt_loc = min(dt_loc_diff, dt_loc_ad)
            dt = min(dt, dt_loc)
    return dt             
        
def solve(params, init_function, **kwargs):
    """Applique la méthode des volumes finis pour le problème défini par les paramètres `params`
    `init_function` est la fonction avec laquelle on initialise les données pour la simulation
    Sauvegarde l'évolution à la fréquence fixée dans les paramètres
    Sauvegarde en calculant *in situ* les énergies cinétiques
    """
    # Initialisation des données
    if params is not None:
        U_old, T0 = func_dict[init_function](params, **kwargs)
        time = []
        ekin_x = []
        ekin_y = []
        t = 0
    else:
        U_old, T0, time, ekin_x, ekin_y, t, params = func_dict[init_function](**kwargs)
    U = np.ones_like(U_old)

    # Discretisation de l'espace
    nx = params[p_nx]
    ny = params[p_ny]    
    dx = params[p_Lx] / nx
    dy = params[p_Ly] / ny   

    # Paramètres pour les sorties
    total_zeros = int(np.log10(params[p_T_end] / params[p_T_io])) + 1
    i = int(t / params[p_T_io])
    t0 = t

    # Enregistre les conditions initiales de la simulation
    if i == 0:
        save_u(U_old, params, params[p_out] + "save_" + "0"*(total_zeros))
        time.append(t)
        ekinx = U_old[:, :, i_momx] ** 2 / U_old[:, :, i_mass]
        ekin_x.append(np.sum(ekinx))
        ekiny = U_old[:, :, i_momy] ** 2 / U_old[:, :, i_mass]
        ekin_y.append(np.sum(ekiny))

    # Boucle principale
    pbar = tqdm.tqdm(total=100) # Progress bar
    i += 1
    while t < params[p_T_end]:

        dt = compute_dt(U_old, nx, ny, dx, dy, params)

        if i * params[p_T_io] <= t+dt:
            n_zeros = int(np.floor(np.log10(i))) + 1
            save_u(U, params, params[p_out] + "save_" + "0"*(total_zeros - n_zeros) + f"{i}")
            i += 1
            dt = (i-1) * params[p_T_io] - t

            # Calcul in-situ des énergies cinétiques
            time.append(t)
            ekinx = U_old[:, :, i_momx] ** 2 / U_old[:, :, i_mass]
            ekin_x.append(np.sum(ekinx))
            ekiny = U_old[:, :, i_momy] ** 2 / U_old[:, :, i_mass]
            ekin_y.append(np.sum(ekiny))
        
        if t+dt > params[p_T_end]:
            dt = params[p_T_end] - t

        # Updating U
        inside_loop(U, U_old, T0, dt, dx, dy, nx, ny, params)
        
        # Condition aux bords
        if params[p_BC] == 'neumann':
            neumann(U, nx, ny)
        elif params[p_BC] == 'periodic':
            periodic(U, nx, ny)
        elif params[p_BC] == 'reflex':
            U = reflex(U, params)
        elif params[p_BC] == 'closed':
            U = closed(U, params)

        U_old = U.copy()
        t += dt

        pbar.update(100*dt/(params[p_T_end]-t0))
    
    pbar.close()    
    
    # Sauvegarde des énergies
    time = np.array(time)
    ekin_x = np.array(ekin_x)
    ekin_y = np.array(ekin_y)

    labels = ('time', 'ekin x', 'ekin y')
    save(params[p_out] + "energies.h5", (time, ekin_x, ekin_y), params, labels)
    
    # Sauvegarde de la température initiale
    save(params[p_out] + 'init.h5', (T0,), params, ('temperature',))