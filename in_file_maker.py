import numpy as np
import h5py
import solve
import utils

p_gamma = 0 # Pour les paramètres de la simulation
p_N = 1
p_T_end = 2
p_CFL = 3
p_BC = 4
p_freq_out = 5
p_in = 6
p_out = 7


def sod_shock_tube(params):
    """ Créer le fichier de condition initiale pour les paramètres donnés
    """
    # En utilisant les primitives (masse, pression, vitesse )
    Q = np.ones((params[p_N]+2, 3), dtype=float) * np.array([1., 1., 0.])
    Q[(params[p_N]+2)//2:params[p_N]+2, :] *= np.array([0.125, 0.1, 0.])
    

    with h5py.File(params[p_in], 'w') as f:
        input_dset = f.create_dataset('input', (params[p_N]+2, 3), dtype=float)
        input_dset[:] = solve.primitive_into_conservative(Q, params)

if __name__ == '__main__':
    params = utils.init_param('test.ini.txt')
    sod_shock_tube(params)