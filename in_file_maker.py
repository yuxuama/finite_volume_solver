import numpy as np
import utils as u
import matplotlib.pyplot as plt
from os import rename, listdir
import h5py

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
        u.periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        u.neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        u.reflex(Q, params, is_conservative=False)
    elif params[p_BC] == 'closed':
        u.closed(Q, params, is_conservative=False)

    plt.pcolormesh(Q[1:nx+1,1:ny+1,0])
    plt.show()

    U = u.primitive_into_conservative(Q, params)
    
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
        u.periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        u.neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        u.reflex(Q, params, is_conservative=False)
    elif params[p_BC] == 'closed':
        u.closed(Q, params, is_conservative=False)
    
    
    U = u.primitive_into_conservative(Q, params)

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

def rt_instability(params, C):
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
        u.periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        u.neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        u.reflex(Q, params, is_conservative=False)
    elif params[p_BC] == 'closed':
        u.closed(Q, params, is_conservative=False)
    
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
    
    U =  u.primitive_into_conservative(Q, params)
    return U, None

def hydrostatic(params):
    """Créer le fichier des conditions initiales pour un cas hydrostatique
    en fonction des paramètres `params` donnés"""
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
        u.periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        u.neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        u.reflex(Q, params, is_conservative=False)
    elif params[p_BC] == 'closed':
        u.closed(Q, params, is_conservative=False)
    
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
    
    U = u.primitive_into_conservative(Q, params)
    return U, None

def simple_convection(params, gradT, T_grd, rho_grd, C, kx, ky):
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
    Q[:, 1, 1] = u.get_pressure_from_temp(rho_grd, T[:, 1], params) # Pression tout en bas

    a = 2 * params[p_cv] * (params[p_gamma] - 1) / (params[p_g] * dy)

    for i in range(1, nx+1):
        for j in range(2, ny+1):
            x = (i-0.5) * dx
            y = (j-0.5) * dy
            Q[i, j, 3] = C * np.sin(np.pi * kx * x / Lx) * np.sin(np.pi * ky * y / Ly) # Perturbation de la vitesse verticale

            Q[i, j, 0] = Q[i, j-1, 0] * (T[i, j-1] * a - 1) / (1 + T[i, j] * a)
            Q[i, j, 1] = u.get_pressure_from_temp(Q[i, j, 0], T[i, j], params) 

    if params[p_BC] == 'periodic':
        u.periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        u.neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        u.reflex(Q, params, is_conservative=False)
    elif params[p_BC] == 'closed':
        u.closed(Q, params, is_conservative=False)

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
    
    U = u.primitive_into_conservative(Q, params)
    return U, T

def simple_diffusion(params, Tdown, Tup, C, kx, rho_grd):
    """Donne les conditions initiales d'un problème de diffusion basique:
    équilibre hydrostatique + thermostat froid en haut et thermostat chaud en bas
    **Input**
    `Tdown` fixe la température en bas
    `Tup` fixe la température en haut
    `p0` fixe la pression en bas de la boîte
    **Output**
    `U` le tableau des grandeurs conseravatives
    `T` le tableau de la température    
    """
    nx = params[p_nx]
    ny = params[p_ny]
    Ly = params[p_Ly]
    dy = Ly / ny

    # Encode les températures aux bords
    T = np.ones((nx+2, ny+2), dtype=float) * 0.5 * (Tdown + Tup)
    T[:, 0] = Tdown + C * np.sin(np.linspace(0, kx * np.pi, nx+2))
    T[:, ny+1] = Tup

    # En utilisant les primitives (masse, pression, vitesse)
    # Encode l'équilibre hydrostatique

    Q = np.zeros((nx+2, ny+2, 4), dtype=float)
    Q[:, 1, 0] = rho_grd
    Q[:, 1, 1] = u.get_pressure_from_temp(rho_grd, T[:, 1], params)

    a = 2 * params[p_cv] * (params[p_gamma] - 1) / (params[p_g] * dy)

    for i in range(1, nx+1):
        for j in range(2, ny+1):
            Q[i, j, 0] = Q[i, j-1, 0] * (T[i, j-1] * a - 1) / (1 + T[i, j] * a)
            Q[i, j, 1] = u.get_pressure_from_temp(Q[i, j, 0], T[i, j], params)
     
    if params[p_BC] == 'periodic':
        u.periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        u.neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        u.reflex(Q, params, is_conservative=False)
    elif params[p_BC] == 'closed':
        u.closed(Q, params, is_conservative=False)
    
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

    U = u.primitive_into_conservative(Q, params)
    return U, T

def layer(params, gradT, T_grd, rho_grd, center, thickness, gradshear, C, kx, ky):
    """Génère les conditions initiale en suivant le papier 'Extremely long phase transition [...]' 
    `gradT` fixe le gradient de température haut - bas
    `T_grd` fixe la température en bas de la boîte
    `rho_grd` fixe la densité en bas de la boîte
    `start_shear` fixe là où commence le gradient de shear
    `gradshear` fixe le gradient de shear appliqué uniquement sur la partie haute de la boîte
    `C` fixe l'intensité de la perturbation
    `kx` fixe le nombre de maxima du mode spatial de perturbation
    """
    nx = params[p_nx]
    ny = params[p_ny]
    Ly = params[p_Ly]
    Lx = params[p_Lx]
    dy = Ly / ny
    dx = Lx / nx

    # Initialisation de la température
    T = np.ones((nx+2, ny+2)) * T_grd - gradT * dy
    for j in range(1, ny+2):
        T[:, j] = T[:, j-1] + gradT * dy
    T[:, 0] = T_grd
    T[:, ny+1] = T[:, ny]

    # Initialisation des variables de description du fluide
    Q = np.zeros((nx+2, ny+2, 4), dtype=float)
    Q[:, 1, 0] = rho_grd
    Q[:, 1, 1] = u.get_pressure_from_temp(rho_grd, T[:, 1], params)

    a = 2 * params[p_cv] * (params[p_gamma] - 1) / (params[p_g] * dy)
    for i in range(1, nx+1):
        for j in range(2, ny+1):
            y = (j-0.5) * dy
            x = (i-0.5) * dx
            Q[i, j, 0] = Q[i, j-1, 0] * (T[i, j-1] * a - 1) / (1 + T[i, j] * a)
            Q[i, j, 1] = u.get_pressure_from_temp(Q[i, j, 0], T[i, j], params)
            Q[i, j, 3] =  C * np.sin(np.pi * kx * x / Lx) * np.sin(np.pi * ky * y / Ly)
            
            if y > center - thickness and y < center:
                Q[i, j, 2] = gradshear * (y - center + thickness) 
            elif y >= center and y < center + thickness:
                Q[i, j, 2] = gradshear * (center + thickness - y)       

    if params[p_BC] == 'periodic':
        u.periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        u.neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        u.reflex(Q, params, is_conservative=False)
    elif params[p_BC] == 'closed':
        u.closed(Q, params, is_conservative=False)

    # Plot
    y = np.arange(0, ny+2)
    potT = u.get_potential_temp(Q[:, :, 1], Q[:, :, 0], params)
    potTshear = u.get_modified_potential_temp(Q[:, :, 1], Q[:, :, 0], Q[:, :, 2], params, kx, ky, params[p_ht] - kx*kx*params[p_k])

    #print(grad_shear_th(Q, params, kx, ky, start_shear))

    fig, ax = plt.subplots(1, 5, figsize=(19, 4))
    fig.suptitle("Conditions initiales")
    ax[0].pcolormesh(Q[:, :, 0].T)
    ax[1].pcolormesh(Q[:, :, 1].T)
    ax[2].pcolormesh(Q[:, :, 2].T)
    ax[3].pcolormesh(Q[:, :, 3].T)
    ax[4].plot(np.mean(potTshear, axis=0), y , '--g', label="Température potentielle modifiée")
    ax[4].plot(np.mean(potT, axis=0), y, 'b', label="Température potentielle")
    ax[0].set_title("Densité")
    ax[1].set_title("Pression")
    ax[2].set_title("Vitesse x")
    ax[3].set_title("Vitesse y")
    ax[4].set_title("Températures")
    ax[4].legend()
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    ax[2].set_aspect('equal', adjustable='box')
    ax[3].set_aspect('equal', adjustable='box')
    plt.show()

    U = u.primitive_into_conservative(Q, params)
    return U, T

def diffusive_layer(params, T_up, T_grd, rho_grd, center, thickness, gradshear, C, kx, ky, shock_limit=0):
    """Créer les conditions initiales pour un problème de confinement de convection par diffusion
    `params` ce sont les paramètres de la simulation définis par le fichier .ini
    `T_up` Température en haut de la boîte
    `T_grd` Température en bas de la boîte
    `rho_grd` Densité volumique de fluide en bas de la boîte
    `center` Centre de la barrière de shear horizontale
    `thickness` DEmi-hauter de la barrière de shear
    `gradshear` gradient de shear dans la barrière de shear
    `C` amplitude de la perturbation
    `kx` Nombre de maxima de la perturbation sinusoïdale selon x
    `ky` Nombre de maxima de la perturbation sinusoïdale selon y
    `shock_limit` Tolérance sur l'amplitude de l'onde de shock issue de la discontinuité de température (si =0 pas pris en compte)
    """
    nx = params[p_nx]
    ny = params[p_ny]
    Ly = params[p_Ly]
    Lx = params[p_Lx]
    dy = Ly / ny
    dx = Lx / nx

    # Initialisation de la température
    T = np.ones((nx+2, ny+2)) * T_up
    T[:, 0] = T_grd
    T[:, ny+1] = T_up
    # On ajoute une petite zone de gradient thermique pour permettre la 'continuité' de la température
    # On cherche à éviter les gros shock
    gradT_cible = -shock_limit / (params[p_k] * params[p_cv] * ny) # Car odg perturbation = k * delta T * cv
    
    if shock_limit > 0:
        if T_grd + gradT_cible * ny > T_up:
            gradT_cible = (T_up - T_grd) / ny
            print("\n/!\ WARNING: l'onde de choc risque d'être trop puissante par rapport à la perturbation")
        print("DEBUG: Valeur prise pour le gradient de température:", gradT_cible, end='\n\n')
        ntot = np.abs((T_up - T_grd) / gradT_cible)
        ntot = int(ntot)
        for j in range(1, ntot+1):
            T[:, j] = T[:, j-1] + gradT_cible

    # Initialisation des variables de description du fluide
    Q = np.zeros((nx+2, ny+2, 4), dtype=float)
    Q[:, 1, 0] = rho_grd
    Q[:, 1, 1] = u.get_pressure_from_temp(rho_grd, T[:, 1], params)
    x = (np.arange(nx+2) - 0.5) * dx
    Q[:, 1, 3] =  C * np.sin(np.pi * kx *  x/ Lx) * np.sin(np.pi * ky * 0.5 * dy / Ly - 0.5 * np.pi)


    a = 2 * params[p_cv] * (params[p_gamma] - 1) / (params[p_g] * dy)
    for i in range(1, nx+1):
        for j in range(2, ny+1):
            y = (j-0.5) * dy
            x = (i-0.5) * dx
            Q[i, j, 0] = Q[i, j-1, 0] * (T[i, j-1] * a - 1) / (1 + T[i, j] * a)
            Q[i, j, 1] = u.get_pressure_from_temp(Q[i, j, 0], T[i, j], params)
            Q[i, j, 3] =  C * np.sin(np.pi * kx * x / Lx) * np.sin(np.pi * ky * y / Ly - 0.5 * np.pi) * np.heaviside(0.5*(Ly / ky) - y, 0)
            
            if y > center - thickness and y < center:
                Q[i, j, 2] = gradshear * (y - center + thickness) 
            elif y >= center and y < center + thickness:
                Q[i, j, 2] = gradshear * (center + thickness - y)       

    if params[p_BC] == 'periodic':
        u.periodic(Q, nx, ny)
    elif params[p_BC] == 'neumann':
        u.neumann(Q, nx, ny)
    elif params[p_BC] == 'reflex':
        u.reflex(Q, params, is_conservative=False)
    elif params[p_BC] == 'closed':
        u.closed(Q, params, is_conservative=False)

    # Plot
    y = np.arange(ny+2)

    fig, ax = plt.subplots(1, 5, figsize=(19, 4))
    fig.suptitle("Conditions initiales")
    ax[0].pcolormesh(Q[:, :, 0].T)
    ax[1].pcolormesh(Q[:, :, 1].T)
    ax[2].pcolormesh(Q[:, :, 2].T)
    ax[3].pcolormesh(Q[:, :, 3].T)
    ax[4].plot(np.mean(T, axis=0), y)
    ax[0].set_title("Densité")
    ax[1].set_title("Pression")
    ax[2].set_title("Vitesse x")
    ax[3].set_title("Vitesse y")
    ax[4].set_title("Profil de Température")
    ax[4].set(xlabel="Température", ylabel="$y$")
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    ax[2].set_aspect('equal', adjustable='box')
    ax[3].set_aspect('equal', adjustable='box')
    plt.show()

    U = u.primitive_into_conservative(Q, params)
    return U, T

def resume_simulation(dirpath, extra_time):
    """Continue la simulation avec les mêmes paramètres que ceux contenus dans le répertoire
    `dirpath` en partant du dernier fichier
    """
    files = [dirpath + f for f in listdir(dirpath)]
    files.sort()

    # Extraction de la dernière frame simulée
    print("Reprise de la simulation à partir du fichier:", files[-1])
    last_frame = h5py.File(files[-1], 'r')
    energies = h5py.File(files[0], 'r')
    f_T0 = h5py.File(files[1], 'r')

    # Extraction des paramètres liés à la simulation
    params = u.extract_parameter(last_frame['metadata'])
    nx = params[p_nx]
    ny = params[p_ny]
    t_old = params[p_T_end]

    print("Les paramètres étaient:")
    params_struct = u.param_struct
    for i in range(len(params_struct)):
        print(params_struct[i][0], "=", params[i])
    print("Le nouveau temps de fin est: ", params[p_T_end] + extra_time)

    # Modification des fichiers existants pour que les métadonnées et les noms de fichier collent
    nzero_old = int(np.log10(params[p_T_end] / params[p_T_io]))
    nzero = int(np.log10((params[p_T_end] + extra_time)/params[p_T_io]))    
    new_params = []
    for p in range(len(params_struct)):
        if p == p_T_end:
            new_params.append(params[p] + extra_time)
        else:
            new_params.append(params[p])
    params = tuple(new_params)

    zero_diff = nzero - nzero_old
    for i in range(2, len(files)):
        #f = h5py.File(files[i], 'w')
        #u.create_all_attribute(f['metadata'], params)
        if zero_diff > 0:
            rename(files[i], files[i][0:len(dirpath + 'save_')] + "0"*zero_diff + files[i][len(dirpath + 'save_'):])

    # Construction de U et des énergies
    U = np.ones((nx+2, ny+2, 4))
    U[1:nx+1, 1:ny+1, 0] = last_frame['rho'][:].T
    U[1:nx+1, 1:ny+1, 1] = last_frame['momentum x'][:].T
    U[1:nx+1, 1:ny+1, 2] = last_frame['momentum y'][:].T
    U[1:nx+1, 1:ny+1, 3] = last_frame['energy'][:].T
    T0 = f_T0['temperature'][:]

    if params[p_BC] == 'neumann':
        u.neumann(U, nx, ny)
    elif params[p_BC] == 'periodic':
        u.periodic(U, nx, ny)
    elif params[p_BC] == 'reflex':
        U = u.reflex(U, params)
    elif params[p_BC] == 'closed':
        U = u.closed(U, params)

    time = list(energies['time'][:])
    ekin_x = list(energies['ekin x'][:])
    ekin_y = list(energies['ekin y'][:])

    return U, T0, time, ekin_x, ekin_y, t_old, params