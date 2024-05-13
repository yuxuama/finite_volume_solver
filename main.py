import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import h5py

# Paramètres du fluide

GAMMA = 1.4
L = 1
N = 400

# Paramètre de la simulation

dx = L / N
i_mass = 0 # Pour les tableaux de grandeurs conservatives
i_mom = 1
i_erg = 2

j_mass = 0 # Pour les tableaux de grandeurs primitives
j_press = 1
j_speed = 2

# Utils

@njit
def get_speed(U=np.ndarray, i=int):
    """Renvoie la vitesse du fluide dans la case i"""
    return U[i, i_mom] / U[i, i_mass]

@njit
def get_pressure(U=np.ndarray, i=int):
    """Renvoie la pression du fluide dans la case i"""
    erg_kin = 0.5 * U[i, i_mom] ** 2 / U[i, i_mass]
    erg_intern = U[i, i_erg] - erg_kin
    return (GAMMA - 1) * erg_intern

@njit
def get_sound_speed(U=np.ndarray, i=int):
    """Renvoie la vitesse du son dans la case de fluide i"""    
    return np.sqrt(GAMMA * get_pressure(U, i) / U[i, i_mass])

@njit
def primitive_into_conservative(Q=np.ndarray):
    """Renvoie le tableau des variables conservatives en partant des variables primitives"""
    U = np.zeros_like(Q)
    U[:, i_mass] = Q[:, j_mass]
    U[:, i_mom] = Q[:, j_mass] * Q[:, j_speed]
    U[:, i_erg] = (Q[:, j_press] / (GAMMA - 1)) + U[:, i_mom]**2 / (2 * U[:, i_mass])

    return U

@njit
def conservative_into_primitive(U):
    """Renvoie le tableau des variables primitives en partant des variables conservatives"""
    Q = np.zeros_like(U)
    mask = np.arange(Q.shape[0])
    Q[:, j_mass] = get_speed(U, mask)
    Q[:, j_speed] = get_pressure(U, mask)
    
    return Q


# Méthode des volumes finis
# On se place dans les conditions aux limites de Neumann et dans un problème 1D

def sod_shock_tube():
    """ Renvoie les conditions initiales du problème des tubes de shock de Sod en variables **conservatives**
    Les conditions initiales sont celles de la pages Wikipedia
    """
    # En utilisant les primitives (masse, pression, vitesse )
    Q = np.ones((N+2, 3), dtype=float) * np.array([1., 1., 0.])
    Q[(N+2)//2:N+2, :] *= np.array([0.125, 0.1, 0.])
    
    # Axe des x

    x = np.linspace(0, L, N)

    return x, primitive_into_conservative(Q)

@njit
def compute_flux(U, i):
    """Renvoie le flux à l'interface entre les case i-1 et i (interface gauche)
    On utilise le solveur de Riemann"""
    
    ul = get_speed(U, i-1)
    ur = get_speed(U, i) 
    pl = get_pressure(U, i-1)
    pr = get_pressure(U, i)

    # Paramètre de couplage vitesse pression
    a = 1.1 * max(U[i-1, i_mass] * get_sound_speed(U, i-1), U[i, i_mass] * get_sound_speed(U, i))

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

def solve(U_i, interval=int):
    """Résout le problème du tube de Sod grâce à la méthode des volumes finis sur `interval` deltas de temps
    en utilisant les conditions initiales données par U_i"""
    
    # tableaux pour sauvegarder l'état
    
    solve = np.zeros((interval+1, N+2, 3), dtype=float)
    solve[0] = U_i
    time = np.zeros(interval+1, dtype=float)

    mask = np.arange(N+2) # Permet de condenser les écritures plus tard

    for t in range(1, interval):
        
        U = solve[t-1]

        max_speed_info = np.max(np.abs(get_speed(U, mask)) + get_sound_speed(U, mask))
        cfl = 0.9
        dt = cfl * dx / max_speed_info
        time[t] = time[t-1] + dt

        for i in range(1, N+1):
            solve[t, i] = U[i] - (dt/dx) * (compute_flux(U, i+1) - compute_flux(U, i))
        solve[t, 0] = solve[t, 1]
        solve[t, N+1] = solve[t, N]

    return time, solve

if __name__ == '__main__':

    x, initial = sod_shock_tube()
    time, data = solve(initial, 300)

    for i in range(0, 300, 30):
        plt.plot(x, data[i, 1:N+1, i_mass], label=round(time[i], 3))
    
    plt.xlabel("$x$")
    plt.ylabel("Densité")
    plt.legend()
    plt.show()