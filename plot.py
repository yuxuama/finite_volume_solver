import numpy as np
import matplotlib.pyplot as plt
import h5py


i_mass = 0 # Pour les tableaux de grandeurs conservatives
i_mom = 1
i_erg = 2

p_gamma = 0 # Pour les paramètres de la simulation
p_N = 1
p_T_end = 2
p_CFL = 3
p_BC = 4
p_freq_out = 5
p_in = 6
p_out = 7

def plot_density(params, out=True):
    """Affiche la densité"""

    # Axe des abscisse
    x = np.linspace(0, 1, params[p_N])

    # Axe des ordonnées
    if out:
        data = h5py.File(params[p_out], 'r')['output']
    else:
        data = h5py.File(params[p_in], 'r')['input']

    plt.plot(x, data[1:params[p_N]+1, i_mass])
    plt.ylabel("Densité")
    plt.xlabel('$x$')
    plt.show()


    