import numpy as np
import matplotlib.pyplot as plt
import h5py

i_mass = 0 # Pour les tableaux de grandeurs conservatives
i_mom = 1
i_erg = 2

def plot_density(filepath):
    """Affiche la densité dans les datas du fichier HDF5 `filepath`"""

    # Load file
    f = h5py.File(filepath, 'r')
    dset = f['data']

    # Extraction d'information
    N = dset.attrs.get("N")
    time = dset.attrs.get("T end")
    name = dset.attrs.get("name")

    # Axe des abscisse
    x = np.linspace(0, 1, N)

    # Axe des ordonnées

    plt.title("{0} (Densité) @ t = {1} s".format(name, time))
    plt.plot(x, dset[1:N+1, i_mass])
    plt.ylabel("Densité")
    plt.xlabel('$x$')
    plt.show()

def plot_speed(filepath):
    """Affiche la vitesse dans les datas du fichier HDF5 `filepath`"""

    # Load file
    f = h5py.File(filepath, 'r')
    dset = f['data']

    # Extraction d'information
    N = dset.attrs.get("N")
    time = dset.attrs.get("T end")
    name = dset.attrs.get("name")

    # Axe des abscisse
    x = np.linspace(0, 1, N)

    # Axe des ordonnées

    plt.title("{0} (Densité) @ t = {1} s".format(name, time))
    plt.plot(x, dset[1:N+1, i_mass])
    plt.ylabel("Densité")
    plt.xlabel('$x$')
    plt.show()

def plot_pressure(filepath):
    """Affiche la pression dans les datas du fichier HDF5 `filepath`"""

    # Load file
    f = h5py.File(filepath, 'r')
    dset = f['data']

    # Extraction d'information
    N = dset.attrs.get("N")
    time = dset.attrs.get("T end")
    name = dset.attrs.get("name")

    # Axe des abscisse
    x = np.linspace(0, 1, N)

    # Axe des ordonnées

    plt.title("{0} (Densité) @ t = {1} s".format(name, time))
    plt.plot(x, dset[1:N+1, i_mass])
    plt.ylabel("Densité")
    plt.xlabel('$x$')
    plt.show()

def plot_all_primitive(filname):
    pass
    