import numpy as np
from utils import get_pressure_from_temp, periodic, reflex, neumann, primitive_into_conservative
import matplotlib.pyplot as plt

p_gamma = 0 # Pour le tuple des paramètres
p_g = 1
p_ht = 2
p_k = 3
p_cv = 4
p_nx = 5
p_ny = 6
p_Lx = 7
p_Ly = 8
p_T_end = 9
p_CFL = 10
p_BC = 11
p_T_io = 12
p_name = 13
p_out = 14

def sod_shock_tube(params, direction):
    """ Créer le fichier de condition initiale pour les paramètres donnés
    pour le problème du tube de Sod 2D avec une orientation donnée par `direction`
    """

    nx = params[p_nx]
    ny = params[p_ny]
    Lx = params[p_Lx]
    Ly = params[p_Ly] 

    # En utilisant les primitives (masse, pression, vitesse )

    Q = np.ones((nx+2, ny+2, 4), dtype=float)

    high = np.array([1., 1., 0., 0.])
    low =  np.array([0.125, 0.1, 0., 0.])

    dx = Lx/nx
    dy = Ly/ny
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            dot_product = direction[0] * ((i-1) * dx - Lx/2) + direction[1] * ((j-1) * dy - Ly/2)
            if dot_product <= 0:
                Q[i, j] *= high
            elif dot_product > 0:
                Q[i, j] *= low 

    if params[p_BC] == 'periodic':
        periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        reflex(Q, params, is_conservative=False)

    plt.pcolormesh(Q[1:nx+1,1:ny+1,0])
    plt.show()

    U = primitive_into_conservative(Q, params)
    
    return U, None
    
def riemann_problem_2d(params):
    """Défini les conditions initiales pour le problème de Riemann 2D"""
    bottom_left = np.array([0.138, 0.029, 1.206, 1.206])
    bottom_right = np.array([0.5323, 0.3, 0.0, 1.206])
    top_left = np.array([0.5323, 0.3, 1.206, 0.0])
    top_right = np.array([1.5, 1.5, 0.0, 0.0])

    nx = params[p_nx]
    ny = params[p_ny]
    Lx = params[p_Lx]
    Ly = params[p_Ly] 

    # En utilisant les primitives (masse, pression, vitesse )

    Q = np.ones((nx+2, ny+2, 4), dtype=float)

    dx = Lx/nx
    dy = Ly/ny
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            x = (i-1)*dx
            y = (j-1)*dy
            if x < 0.8 * Lx:
                if y < 0.8 * Ly:
                    Q[i, j] *= bottom_left
                else:
                    Q[i, j] *= top_left
            else:
                if y < 0.8 * Ly:
                    Q[i, j] *= bottom_right
                else:
                    Q[i, j] *= top_right
            
    if params[p_BC] == 'periodic':
        periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        reflex(Q, params, is_conservative=False)
    
    
    U = primitive_into_conservative(Q, params)

    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle("Conditions initiales")
    ax[0].pcolormesh(U[1:nx+1,1:ny+1, 0])
    ax[1].pcolormesh(U[1:nx+1,1:ny+1, 1])
    ax[2].pcolormesh(U[1:nx+1,1:ny+1, 2])
    ax[3].pcolormesh(U[1:nx+1,1:ny+1, 3])
    ax[0].set_title("Densité")
    ax[1].set_title("Impulsion x")
    ax[2].set_title("Impulsion y")
    ax[3].set_title("Énergie")
    plt.show()

    return U, None

def rt_instability(params, C=0):
    """Créer le fichier des conditions initiale pour les paramètres donnés
    et pour le problème de l'instabilité de Rayleigh-Taylor
    `C` est l'amplitude de la perturbation de l'équilibre hydrostatique
    """
    nx = params[p_nx]
    ny = params[p_ny]
    Lx = params[p_Lx]
    Ly = params[p_Ly]
    dx = Lx / nx
    dy = Ly / ny
    p0 = 2.5

    # En utilisant les primitives (masse, pression, vitesse)

    Q = np.zeros((nx+2, ny+2, 4), dtype=float)
    Q[:, 1, 1] = p0 * np.ones((nx+2,))
    Q[:, 1, 0] = np.ones((nx+2,))
    
    for i in range(1, nx+1):
        x = (i-0.5) * dx
        Q[i, 1, 3] = 0.25 * C * (1 + np.cos(4 * np.pi * (x - 0.5 * Lx))) * (1 + np.cos(3 * np.pi * 0.5 * Ly))
        for j in range(2, ny+1):
            y = (j-0.5) * dy
            if y < Ly / 2:
                Q[i, j, 0] = 1
            else:
                Q[i, j, 0] = 2
            Q[i, j, 1] = Q[i, j-1, 1] - dy * params[p_g] * 0.5 * (Q[i, j-1, 0] + Q[i, j, 0])
            Q[i, j, 3] = 0.25 * C * (1 + np.cos(4 * np.pi * (x - 0.5 * Lx))) * (1 + np.cos(3 * np.pi * (y - 0.5 * Ly)))
     
    if params[p_BC] == 'periodic':
        periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        reflex(Q, params, is_conservative=False)
    
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle("Conditions initiales")
    ax[0].pcolormesh(Q[:, :, 0].T)
    ax[1].pcolormesh(Q[:, :, 1].T)
    ax[2].pcolormesh(Q[:, :, 2].T)
    ax[3].pcolormesh(Q[:, :, 3].T)
    ax[0].set_title("Densité")
    ax[1].set_title("Pression")
    ax[2].set_title("Vitesse x")
    ax[3].set_title("Vitesse y")
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    ax[2].set_aspect('equal', adjustable='box')
    ax[3].set_aspect('equal', adjustable='box')
    plt.show()
    
    U =  primitive_into_conservative(Q, params)
    return U, None

def hydrostatic(params):
    """Créer le fichier des conditions initiales pour un cas hydrostatique
    en fonction des paramètres donnés"""
    nx = params[p_nx]
    ny = params[p_ny]
    Lx = params[p_Lx]
    Ly = params[p_Ly]
    dy = Ly / ny
    dx = Lx / nx
    p0 = 2.5

    # En utilisant les primitives (masse, pression, vitesse)

    Q = np.zeros((nx+2, ny+2, 4), dtype=float)
    Q[:, 1, 1] = p0 * np.ones((nx+2,))
    Q[:, 1, 0] = np.ones((nx+2,), dtype=float) * 0.1
    
    for i in range(1, nx+1):
        for j in range(2, ny+1):
            x = (i - 0.5) * dx
            y = (j - 0.5) * dy
            if y < Ly / 2:
                Q[i, j, 0] = 0.1
            else:
                Q[i, j, 0] = 0.2
            Q[i, j, 1] = Q[i, j-1, 1] - dy * params[p_g] * 0.5 * (Q[i, j-1, 0] + Q[i, j, 0])
     
    if params[p_BC] == 'periodic':
        periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        reflex(Q, params, is_conservative=False)
    
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle("Conditions initiales")
    ax[0].pcolormesh(Q[:, :, 0].T)
    ax[1].pcolormesh(Q[:, :, 1].T)
    ax[2].pcolormesh(Q[:, :, 2].T)
    ax[3].pcolormesh(Q[:, :, 3].T)
    ax[0].set_title("Densité")
    ax[1].set_title("Pression")
    ax[2].set_title("Vitesse x")
    ax[3].set_title("Vitesse y")
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    ax[2].set_aspect('equal', adjustable='box')
    ax[3].set_aspect('equal', adjustable='box')
    plt.show()
    
    U = primitive_into_conservative(Q, params)
    return U, None

def simple_convection(params, gradT=0, T_grd=0, rho_grd=1, C=0, kx=0, ky=0):
    """Calcul la condition initiale pour un problème de convection basique
    `gradT` fixe le gradient de température vertical
    `rho_grd` fixe la valeur de la densité en bas de la boîte
    `T_grf` fixe la valeur de la température en bas de la boîte
    `C` est l'amplitude de la perturbation appliquée
    `kx` est la fréquence spatiale selon x normalisée
    `ky` est la fréquence spatiale selon y normalisée    
    """
    nx = params[p_nx]
    ny = params[p_ny]
    Lx = params[p_Lx]
    Ly = params[p_Ly]
    dy = Ly / ny
    dx = Lx / nx

    # Température en fonction de l'altitude (les valeurs aux bords sont aberrantes)
    T = np.ones((nx+2, ny+2)) * T_grd - gradT * dy
    for j in range(1, ny+2):
        T[:, j] = T[:, j-1] + gradT * dy

    # En utilisant les primitives (masse, pression, vitesse)

    Q = np.zeros((nx+2, ny+2, 4), dtype=float)
    Q[:, 1, 0] = rho_grd # Masse tout en bas
    Q[:, 1, 1] = get_pressure_from_temp(rho_grd, T[:, 1], params) # Pression tout en bas

    a = 2 * params[p_cv] * (params[p_gamma] - 1) / (params[p_g] * dy)

    for i in range(1, nx+1):
        for j in range(2, ny+1):
            x = (i-0.5) * dx
            y = (j-0.5) * dy
            Q[i, j, 3] = C * np.sin(np.pi * kx * x / Lx) * np.sin(np.pi * ky * y / Ly) # Perturbation de la vitesse verticale

            Q[i, j, 0] = Q[i, j-1, 0] * (T[i, j-1] * a - 1) / (1 + T[i, j] * a)
            Q[i, j, 1] = get_pressure_from_temp(Q[i, j, 0], T[i, j], params) 

    if params[p_BC] == 'periodic':
        periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        reflex(Q, params, is_conservative=False)

    fig, ax = plt.subplots(1, 5, figsize=(17, 4))
    fig.suptitle("Conditions initiales")
    ax[0].pcolormesh(Q[:, :, 0].T)
    ax[1].pcolormesh(Q[:, :, 1].T)
    ax[2].pcolormesh(Q[:, :, 2].T)
    ax[3].pcolormesh(Q[:, :, 3].T)
    ax[4].plot(T, np.linspace(-dy, Lx+dy, nx+2))
    ax[0].set_title("Densité")
    ax[1].set_title("Pression")
    ax[2].set_title("Vitesse x")
    ax[3].set_title("Vitesse y")
    ax[4].set_title("Profil de température")
    ax[4].set(xlabel='Température', ylabel='$y$')
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    ax[2].set_aspect('equal', adjustable='box')
    ax[3].set_aspect('equal', adjustable='box')
    plt.show()
    
    U = primitive_into_conservative(Q, params)
    return U, T

def simple_diffusion(params, Tdown=0, Tup=0, C=0, kx=0, rho_grd = 1):
    """Donne les conditions initiales d'un problème de diffusion basique:
    équilibre hydrostatique + thermostat froid en haut et thermostat chaud en bas
    Pas de perturbation
    `Tdown` fixe la température en bas
    `Tup` fixe la température en haut
    `p0` fixe la pression en bas de la boîte
    """
    nx = params[p_nx]
    ny = params[p_ny]
    Lx = params[p_Lx]
    Ly = params[p_Ly]
    dy = Ly / ny
    dx = Lx / nx

    # Encode les températures aux bords
    T = np.ones((nx+2, ny+2), dtype=float) * 0.5 * (Tdown + Tup)
    T[:, 0] = Tdown + C * np.sin(np.linspace(0, kx * np.pi, nx+2))
    T[:, ny+1] = Tup

    # En utilisant les primitives (masse, pression, vitesse)
    # Encode l'équilibre hydrostatique

    Q = np.zeros((nx+2, ny+2, 4), dtype=float)
    Q[:, 1, 0] = rho_grd
    Q[:, 1, 1] = get_pressure_from_temp(rho_grd, T[:, 1], params)

    a = 2 * params[p_cv] * (params[p_gamma] - 1) / (params[p_g] * dy)

    for i in range(1, nx+1):
        for j in range(2, ny+1):
            Q[i, j, 0] = Q[i, j-1, 0] * (T[i, j-1] * a - 1) / (1 + T[i, j] * a)
            Q[i, j, 1] = get_pressure_from_temp(Q[i, j, 0], T[i, j], params)
     
    if params[p_BC] == 'periodic':
        periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        reflex(Q, params, is_conservative=False)
    
    fig, ax = plt.subplots(1, 5, figsize=(17, 4))
    fig.suptitle("Conditions initiales")
    ax[0].pcolormesh(Q[:, :, 0].T)
    ax[1].pcolormesh(Q[:, :, 1].T)
    ax[2].pcolormesh(Q[:, :, 2].T)
    ax[3].pcolormesh(Q[:, :, 3].T)
    ax[4].pcolormesh(T.T)
    ax[0].set_title("Densité")
    ax[1].set_title("Pression")
    ax[2].set_title("Vitesse x")
    ax[3].set_title("Vitesse y")
    ax[4].set_title("Température")
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    ax[2].set_aspect('equal', adjustable='box')
    ax[3].set_aspect('equal', adjustable='box')
    ax[4].set_aspect('equal', adjustable='box')
    plt.show()

    U = primitive_into_conservative(Q, params)
    return U, T
