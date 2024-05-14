import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils import get_pressure, get_speed, extract_parameter

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

    # Axes des ordonnées

    mask = np.arange(1, N+1)
    data = get_speed(dset[:], mask)

    plt.title("{0} (Vitesse) @ t = {1} s".format(name, time))
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

    params = extract_parameter(dset)

    # Axe des abscisse
    x = np.linspace(0, 1, N)

    # Axe des ordonnées

    mask = np.arange(1, N+1)
    data = get_pressure(dset[:], mask, params)

    plt.title("{0} (Pression) @ t = {1} s".format(name, time))
    plt.plot(x, data)
    plt.ylabel("Densité")
    plt.xlabel('$x$')
    plt.show()

def plot_all_primitive(filepath):
    """Affiche toutes les grandeurs dans un même graphique"""

    # Load file
    f = h5py.File(filepath, 'r')
    dset = f['data']

    # Extraction d'information

    params = extract_parameter(dset)

    N = dset.attrs.get("N")
    time = dset.attrs.get("T end")
    name = dset.attrs.get("name")

    # Axe des abscisse
    x = np.linspace(0, 1, N)

    # Axe des ordonnées

    mask = np.arange(1, N+1)
    speed = get_speed(dset[:], mask)
    pressure = get_pressure(dset[:], mask, params)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("{0} @ t = {1} s".format(name, time))

    ax[0].plot(x, dset[1:N+1, 0])
    ax[0].set(xlabel="$x$", ylabel="Densité", title="Densité")

    ax[1].plot(x, speed)
    ax[1].set(xlabel="$x$", ylabel="Vitesse", title="Vitesse")
    
    ax[2].plot(x, pressure)
    ax[2].set(xlabel="$x$", ylabel="Pression", title="Pression")

    plt.show()

if __name__ == "__main__":
    plot_all_primitive('./out/sod shock')