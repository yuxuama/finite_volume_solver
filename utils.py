from numba import njit, prange
import numpy as np
import h5py
import os

i_mass = 0 # Pour les tableaux de grandeurs conservatives
i_momx = 1
i_momy = 2
i_erg = 3

j_mass = 0 # Pour les tableaux de grandeurs primitives
j_press = 1
j_speedx = 2
j_speedy = 3

p_gamma = 0 # Pour les tableaux de grandeurs primitives
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
p_freq_out = 12
p_name = 13
p_in = 14
p_out = 15

param_struct = [  ("Gamma", float),
                ("g", float),
                ("ht", float),
                ("k", float),
                ("cv", float),
                ("nx", int),
                ("ny", int),
                ("Lx", float),
                ("Ly", float),
                ("T end", float),
                ("CFL", float),
                ("BC", str),
                ("freq out", float),
                ("name", str),
                ("input name", str),
                ("output dir", str)
]

def init_param(filename):
    """Définit tous les paramètres de la simulation étant donné un fichier txt du bon format"""
    params = [0 for _ in range(len(param_struct))]
    with open(filename, 'r') as f:
        i = 0
        for line in f.readlines():

            header_size = len(param_struct[i][0]) + 1 # On inclut les :

            if param_struct[i][1] is float:
                params[i] = float(line[header_size::])
            elif param_struct[i][1] is str:
                params[i] = line[header_size::].lower().strip()
            elif param_struct[i][1] is int:
                params[i] = int(line[header_size::])
            i+=1
        
    assert params[p_BC] in ['neumann', 'periodic', 'reflex']
    params[p_in] = "./in/" + params[p_in] + '.h5'
    params[p_out] = "./out/" + params[p_out] + '/'
    os.makedirs(params[p_out], exist_ok=True)
    return tuple(params) # Pour que numba fonctionne (objet non modifiable)

def create_all_attribute(dset, params):
    """Met en attribut d'un hdf5 tous les paramètres de la simumation"""
    for i in range(len(param_struct)):
        if param_struct[i][0] in dset.attrs.keys():
            dset.attrs.modify(param_struct[i][0], params[i])
        else:
            dset.attrs.create(param_struct[i][0], params[i])

def extract_parameter(dset):
    """Récupère les paramètres en utilisant les attributs d'un dataset"""
    params = [0 for _ in range(len(param_struct))]
    for i in range(len(param_struct)):
        params[i] = dset.attrs.get(param_struct[i][0])
    return tuple(params)

def save_u(U, params, filepath, masks=None):
    """Sauvegarde les données de U dans un fichier HDF5"""

    nx = params[p_nx]
    ny = params[p_ny]

    if masks is None:
        mask_x = np.arange(1, nx+1) 
        mask_y = np.arange(1, ny+1) 
        mask_x, mask_y = np.meshgrid(mask_x, mask_y)

    f = h5py.File(filepath, "w")
    
    f["x"] = np.linspace(0, params[p_Lx], nx)
    f["y"] = np.linspace(0, params[p_Ly], ny)
    f.create_group('metadata')
    create_all_attribute(f['metadata'], params) # Enregistre en métadonnée les paramètres
    f["rho"] = U[mask_x, mask_y, i_mass]
    f["momentum x"] = U[mask_x, mask_y, i_momx]
    f["momentum y"] = U[mask_x, mask_y, i_momy]
    f["energy"] = U[mask_x, mask_y, i_erg]

    Q = conservative_into_primitive(U, params)

    f["pressure"] = Q[mask_x, mask_y, j_press]
    f["speed x"] = Q[mask_x, mask_y, j_speedx]       
    f["speed y"] = Q[mask_x, mask_y, j_speedy] 

def save(filepath, data, params, label=None):
    """Sauvegarde chaque élément du tuple `data` dans un fichier HDF5
    `data` doit être un tuple de `np.ndarray`
    Si `label` n'est pas `None` alors ce doit être un tuple de même taille que `data` 
    """
    f = h5py.File(filepath, 'w')

    if label is None:
        label = np.arange(len(data))
    for i in range(len(data)):
        f[label[i]] = data[i]
    
    f.create_group("metadata")
    create_all_attribute(f['metadata'], params)

# Conservatives and primitives utils

@njit
def get_speed_x(U):
    """Renvoie la vitesse du fluide selon x
    """
    return U[i_momx] / U[i_mass]

@njit
def get_speed_y(U):
    """Renvoie la vitesse du fluide selon y"""
    return U[i_momy] / U[i_mass]

@njit
def get_speed(U):
    """Renvoie la vitesse du fluide dans la case i"""
    return np.sqrt((get_speed_x(U))**2 + (get_speed_y(U))**2)

@njit
def get_pressure(U, params):
    """Renvoie la pression du fluide dans la case i"""
    erg_kin = 0.5 * (U[i_momx] ** 2 / U[i_mass] + U[i_momy] ** 2 / U[i_mass])
    erg_intern = U[i_erg] - erg_kin
    return (params[p_gamma] - 1) * erg_intern

@njit
def get_pressure_from_temp(rho, T, params):
    """Renvoie la pression calculé à partir de la température"""
    return rho * params[p_cv] * T * (params[p_gamma] - 1)

@njit
def get_temp_from_pressure(press, rho, params):
    """Renvoie la valeur de la température à partir de la pression donnée"""
    return press / (rho * params[p_cv] * (params[p_gamma] - 1))

@njit
def get_temp(U, params):
    """Renvoie la température pour un vecteur d'état de fluide U """
    return get_pressure(U, params) / (U[i_mass] * params[p_cv] * (params[p_gamma] - 1))

@njit
def get_sound_speed(U, params):
    """Renvoie la vitesse du son dans la case de fluide i"""    
    return np.sqrt(params[p_gamma] * get_pressure(U, params) / U[i_mass])

@njit
def primitive_into_conservative(Q, params):
    """Renvoie le tableau des variables conservatives en partant des variables primitives"""
    U = np.zeros_like(Q)
    densite_temp = Q[:, :, j_mass]
    U[:, :, i_mass] = densite_temp
    U[:, :, i_momx] = densite_temp * Q[:, :, j_speedx]
    U[:, :, i_momy] = densite_temp * Q[:, :, j_speedy]
    U[:, :, i_erg] = (Q[:, :, j_press] / (params[p_gamma] - 1)) + U[:,:, i_momx]**2 / (2 * densite_temp) + U[:, :, i_momy]**2 / (2 * densite_temp)

    return U

@njit
def conservative_into_primitive(U, params):
    """Renvoie le tableau des variables primitives en partant des variables conservatives"""
    Q = np.zeros_like(U)
    Q[:, :, j_mass] = U[:, :, i_mass]
    for i in prange(params[p_nx]+2):
        for j in prange(params[p_ny]+2):
            Q[i, j, j_speedx] = U[i, j, i_momx] / U[i, j, i_mass]
            Q[i, j, j_speedy] = U[i, j, i_momy] / U[i, j, i_mass]
            Q[i, j, j_press] = get_pressure(U[i, j], params)
    
    return Q

# Boundary conditions

@njit
def neumann(U, nx, ny):
    """Modifie U pour qu'il vérifie les conditions aux limites de Neumann"""
    U[0, :, :] = U[1, :, :] 
    U[nx+1, :, :] = U[nx, :, :]
    U[:, 0, :] = U[:, 1, :]
    U[:, ny+1, :] = U[:, ny, :]

@njit
def periodic(U, nx, ny):
    """Modifie U pour qu'il vérifie les conditions aux bords périodique """
    U[0, :, :] = U[nx, :, :]
    U[nx + 1, :, :] = U[1, :, :]
    U[:, 0, :] = U[:, ny, :]
    U[:, ny+1, :] = U[:, 1, :]

@njit
def reflex(Q, params, is_conservative=True):
    """Modifie U pour que le tableaux vérifie les conditions réflexives
    Périodique selon les x
    Blocage selon y (fermeture) vérifiant l'équilibre hydro
    `is_conservative` permet de traiter le cas où la variable passée et le vecteur des grandeurs conservatives
    """
    nx = params[p_nx]
    ny = params[p_ny]
    Ly = params[p_Ly]
    dy = Ly/ny

    if is_conservative:
        Q = conservative_into_primitive(Q, params)
    # Périodique selon les x
    Q[0, :, :] = Q[nx, :, :]
    Q[nx + 1, :, :] = Q[1, :, :]
    # Fermeture de la boîte en respectant l'équilibre
    Q[:, 0, j_mass] = Q[:, 1, j_mass] # Masse
    Q[:, ny+1, j_mass] = Q[:, ny, j_mass]
    Q[:, 0, j_speedx] = Q[:, 1, j_speedx] # Vitesse selon x
    Q[:, ny+1, j_speedx] = Q[:, ny, j_speedx]
    Q[:, 0, j_speedy] = - Q[:, 1, j_speedy] # Vitesse selon y
    Q[:, ny+1, j_speedy] = - Q[:, ny, j_speedy]
    Q[:, 0, j_press] = Q[:, 1, j_press] + params[p_g] * dy * Q[:, 1, j_mass] # Pression (équilibre hydro)
    Q[:, ny+1, j_press] = Q[:, ny, j_press] - params[p_g] * dy * Q[:, ny, j_mass]

    if is_conservative:
        return primitive_into_conservative(Q, params)


