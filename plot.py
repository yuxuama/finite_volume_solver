import numpy as np
import matplotlib.pyplot as plt
import h5py


i_mass = 0 # Pour les tableaux de grandeurs conservatives
i_mom = 1
i_erg = 2

def plot_density(filepath):
    """Affiche la densité"""

    # Load file
    f = h5py.File(filepath, 'r')
    data = f['main'][:]

    # Extraction d'information
    N = data.shape[0] - 2

    # Axe des abscisse
    x = np.linspace(0, 1, N)

    # Axe des ordonnées

    plt.plot(x, data[1:N+1, i_mass])
    plt.ylabel("Densité")
    plt.xlabel('$x$')
    plt.show()


    