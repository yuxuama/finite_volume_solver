import numpy as np
import matplotlib.pyplot as plt
import h5py


def plot_density(filepath, ax=None, **kwargs):
    """Affiche la densité dans les datas du fichier HDF5 `filepath`
    `ax` permet de plot sur une autre figure
    """

    # Load file
    f = h5py.File(filepath, 'r')
    dset_x = f['x']

    # Extraction d'information
    time = dset_x.attrs.get("T end")
    name = dset_x.attrs.get("name")

    # Plot
    plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot = True

    ax.set_title("{0} (Densité) @ t = {1} s".format(name, time))
    ax.plot(dset_x[:], f["rho"][:], **kwargs)
    ax.set_ylabel("Densité $m^{-3}$")
    ax.set_xlabel('$x$')

    if plot:
        plt.show()


def plot_speed(filepath, ax=None, **kwargs):
    """Affiche la densité dans les datas du fichier HDF5 `filepath`
    `ax` permet de plot sur une autre figure
    """

    # Load file
    f = h5py.File(filepath, 'r')
    dset_x = f['x']

    # Extraction d'information
    time = dset_x.attrs.get("T end")
    name = dset_x.attrs.get("name")

    # Plot
    plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot = True

    ax.plot(dset_x[:], f["speed"][:], **kwargs)
    ax.set_title("{0} (Vitesse) @ t = {1} s".format(name, time))
    ax.set_ylabel("Vitesse ($m.s^{-1}$)")
    ax.set_xlabel('$x$')
    
    if plot:
        plt.show()


def plot_pressure(filepath, ax=None, **kwargs):
    """Affiche la densité dans les datas du fichier HDF5 `filepath`
    `ax` permet de plot sur une autre figure
    """

    # Load file
    f = h5py.File(filepath, 'r')
    dset_x = f['x']

    # Extraction d'information
    time = dset_x.attrs.get("T end")
    name = dset_x.attrs.get("name")

    # Plot
    plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot = True
    
    ax.plot(dset_x[:], f["pressure"][:], **kwargs)
    ax.set_title("{0} (Pression) @ t = {1} s".format(name, time))
    ax.set_ylabel("Pression $Pa$")
    ax.set_xlabel('$x$')

    if plot:
        plt.show()

def plot_all_primitive(filepath):
    """Affiche toutes les grandeurs dans un même graphique"""

    # Load file
    f = h5py.File(filepath, 'r')
    dset_x = f['x']

    N = dset_x.attrs.get("N")
    time = dset_x.attrs.get("T end")
    name = dset_x.attrs.get("name")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("{0} @ t = {1} s".format(name, time))

    plot_density(filepath, ax=ax[0])
    ax[0].set( xlabel="$x$", ylabel="Densité $(m^{-3})$", title="Densité")
    plot_speed(filepath, ax=ax[1])
    ax[1].set(xlabel="$x$", ylabel="Vitesse ($m.s^{-1}$)", title="Vitesse")
    plot_pressure(filepath, ax=ax[2])
    ax[2].set(xlabel="$x$", ylabel="Pression (Pa)", title="Pression")

    plt.show()

if __name__ == "__main__":
    file = "./out/sod shock.h5"
    plot_all_primitive(file)