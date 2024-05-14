import h5py
from numba import njit

i_mass = 0 # Pour les tableaux de grandeurs conservatives
i_mom = 1
i_erg = 2

j_mass = 0 # Pour les tableaux de grandeurs primitives
j_press = 1
j_speed = 2

p_gamma = 0 # Pour les tableaux de grandeurs primitives
p_N = 1
p_T_end = 2
p_CFL = 3
p_BC = 4
p_freq_out = 5
p_name = 6
p_in = 7
p_out = 8

param_struct = [  ("Gamma", float),
                ("N", int),
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
    params = [0 for i in range(9)]
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
    params[p_in] = "./in/" + params[p_in]
    params[p_out] = "./out/" + params[p_out]
    return tuple(params) # Pour que numba fonctionne (objet non modifiable)

def create_all_attribute(hdf_dset, params):
    """Met en attribut d'un hdf5 tous les paramètres de la simumation"""
    for i in range(7):
        hdf_dset.attrs.create(param_struct[i][0], params[i])

# Conservatives and primitives utils

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


# Test

if __name__ == '__main__':
    print(init_param('./test.ini.txt'))