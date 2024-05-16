import numpy as np
import h5py
from utils import init_param, create_all_attribute, primitive_into_conservative
import matplotlib.pyplot as plt

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
                count += 1
                Q[i, j] *= high
            elif dot_product > 0:
                Q[i, j] *= low 

    if params[p_BC] == 'periodic':
        Q[0] = Q[nx]
        Q[nx + 1] = Q[1]
        Q[:, 0] = Q[:, ny]
        Q[:, ny+1] = Q[:, 1]
    elif params[p_BC] == 'neumann':
        Q[0] = Q[1]
        Q[nx + 1] = Q[nx]
        Q[:, 0] = Q[:, 1]
        Q[:, ny+1] = Q[:, ny]

    with h5py.File(params[p_in], 'w') as f:
        f['data'] = primitive_into_conservative(Q, params)
        f.create_group('metadata')
        create_all_attribute(f['metadata'], params)
    

def riemann_problem_2d(params):
    """Défini les conditions initiales pour le problème de Riemann 2D"""
    bottom_left = np.array([0.138, 1.206, 1.206, 0.029])
    bottom_right = np.array([0.5323, 0.0, 1.206, 0.3])
    top_left = np.array([0.5323, 1.206, 0.0, 0.3])
    top_right = np.array([1.5, 0.0, 0.0, 1.5])

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
        Q[0] = Q[nx]
        Q[nx + 1] = Q[1]
        Q[:, 0] = Q[:, ny]
        Q[:, ny+1] = Q[:, 1]
    elif params[p_BC] == 'neumann':
        Q[0] = Q[1]
        Q[nx + 1] = Q[nx]
        Q[:, 0] = Q[:, 1]
        Q[:, ny+1] = Q[:, ny]
    
    
    with h5py.File(params[p_in], 'w') as f:
        U = primitive_into_conservative(Q, params)
        f['data'] = U

        fig, ax = plt.subplots(1, 4, figsize=(15, 4))
        ax[0].pcolormesh(U[1:nx+1,1:ny+1, 0])
        ax[1].pcolormesh(U[1:nx+1,1:ny+1, 1])
        ax[2].pcolormesh(U[1:nx+1,1:ny+1, 2])
        ax[3].pcolormesh(U[1:nx+1,1:ny+1, 3])
        plt.show()

        f.create_group('metadata')
        create_all_attribute(f['metadata'], params)

# Depreciated
def two_rarefaction(params):
    """Créer le fichier de condition initiale pour les paramètres donnés
    pour le problème de la 2-raréfaction
    """
    
    N = params[p_nx]

    Q = np.ones((N+2, 3)) * np.array([1., 0.4, -2.])
    Q[(N+2)//2:N+2, :] *= np.array([1., 1., -1.])

    if params[p_BC] == 'periodic':
        Q[0] = Q[N]
        Q[N + 1] = Q[1]

    with h5py.File(params[p_in], 'w') as f:
        input_dset = f.create_dataset('data', (N+2, 3), dtype=float)
        input_dset[:] = primitive_into_conservative(Q, params)
        create_all_attribute(input_dset, params)

if __name__ == '__main__':
    params = init_param('test.ini.txt')
    sod_shock_tube(params)