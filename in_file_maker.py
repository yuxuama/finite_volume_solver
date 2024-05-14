import numpy as np
import h5py
import solve
from utils import init_param, create_all_attribute

p_gamma = 0
p_N = 1
p_T_end = 2
p_CFL = 3
p_BC = 4
p_freq_out = 5
p_name = 6
p_in = 7
p_out = 8



def sod_shock_tube(params):
    """ Créer le fichier de condition initiale pour les paramètres donnés
    pour le problème du tube de Sod
    """

    N = params[p_N]

    # En utilisant les primitives (masse, pression, vitesse )
    Q = np.ones((N+2, 3), dtype=float) * np.array([1., 1., 0.])
    Q[(N+2)//2:N+2, :] *= np.array([0.125, 0.1, 0.])
    
    if params[p_BC] == 'periodic':
        Q[0] = Q[N]
        Q[N + 1] = Q[1]

    with h5py.File(params[p_in], 'w') as f:
        input_dset = f.create_dataset('data', (N+2, 3), dtype=float)
        input_dset[:] = solve.primitive_into_conservative(Q, params)
        create_all_attribute(input_dset, params)
    


def two_rarefaction(params):
    """Créer le fichier de condition initiale pour les paramètres donnés
    pour le problème de la 2-raréfaction
    """
    
    N = params[p_N]

    Q = np.ones((N+2, 3)) * np.array([1., 0.4, -2.])
    Q[(N+2)//2:N+2, :] *= np.array([1., 1., -1.])

    if params[p_BC] == 'periodic':
        Q[0] = Q[N]
        Q[N + 1] = Q[1]

    with h5py.File(params[p_in], 'w') as f:
        input_dset = f.create_dataset('data', (N+2, 3), dtype=float)
        input_dset[:] = solve.primitive_into_conservative(Q, params)
        create_all_attribute(input_dset, params)

if __name__ == '__main__':
    params = init_param('test.ini.txt')
    sod_shock_tube(params)