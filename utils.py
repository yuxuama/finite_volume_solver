from numba import njit, jit
import numpy as np
import h5py

i_mass = 0 # Pour les tableaux de grandeurs conservatives
i_momx = 1
i_momy = 2
i_erg = 3

j_mass = 0 # Pour les tableaux de grandeurs primitives
j_press = 1
j_speedx = 2
j_speedy = 3

p_gamma = 0 # Pour les tableaux de grandeurs primitives
p_nx = 1
p_ny = 2
p_Lx = 3
p_Ly = 4
p_T_end = 5
p_CFL = 6
p_BC = 7
p_freq_out = 8
p_name = 9
p_in = 10
p_out = 11

param_struct = [  ("Gamma", float),
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
                ("output name", str)
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
        
    assert params[p_BC] in ['neumann', 'periodic']
    params[p_in] = "./in/" + params[p_in] + '.h5'
    params[p_out] = "./out/" + params[p_out] + '.h5'
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

def save(U, params, filepath=None, masks=None):
    """Sauvegarde les données de U dans un fichier HDF5"""

    nx = params[p_nx]
    ny = params[p_ny]

    if masks is None:
        mask_x = np.arange(1, nx+1) 
        mask_y = np.arange(1, ny+1) 
        mask_x, mask_y = np.meshgrid(mask_x, mask_y)

    if filepath is not None:
        f = h5py.File(filepath, "w")
    else:
        f =  h5py.File(params[p_out], "w")
    
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
    f["speed y"] = Q[mask_x, mask_y, j_speedx] 

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
def get_sound_speed(U, params):
    """Renvoie la vitesse du son dans la case de fluide i"""    
    return np.sqrt(params[p_gamma] * get_pressure(U, params) / U[i_mass])

def primitive_into_conservative(Q, params):
    """Renvoie le tableau des variables conservatives en partant des variables primitives"""
    U = np.zeros_like(Q)
    densite_temp = Q[:, :, j_mass]
    U[:, :, i_mass] = densite_temp
    U[:, :, i_momx] = densite_temp * Q[:, :, j_speedx]
    U[:, :, i_momy] = densite_temp * Q[:, :, j_speedy]
    U[:, :, i_erg] = (Q[:, :, j_press] / (params[p_gamma] - 1)) + U[:,:, i_momx]**2 / (2 * densite_temp) + U[:, :, i_momy]**2 / (2 * densite_temp)

    return U

def conservative_into_primitive(U, params):
    """Renvoie le tableau des variables primitives en partant des variables conservatives"""
    Q = np.zeros_like(U)
    Q[:, :, j_mass] = U[:, :, i_mass]
    Q[:, :, j_speedx] = get_speed_x(U.T).T
    Q[:, :, j_speedy] = get_speed_y(U.T).T
    Q[:, :, j_press] = get_pressure(U.T, params).T
    
    return Q
