import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils import extract_parameter

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

def plot_density_slice(filepath, slice_index, axis, ax=None, **kwargs):
    """Affiche une tranche de la densité issue des datas du fichier HDF5 `filepath` selon l'axe `axis` et pour l'indice `slice_index`
    `ax` permet de plot sur une autre figure
    """

    if axis == 0:
        axis = 'x'
    elif axis == 1:
        axis = 'y'

    # Load file
    f = h5py.File(filepath, 'r')
    meta = f['metadata']
    dset = f[axis]

    # Extraction d'information
    time = meta.attrs.get("T end")
    name = meta.attrs.get("name")

    # Plot
    plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot = True
    
    if axis == 'x':
        data = f["rho"][slice_index]
    elif axis == 'y':
        data = f["rho"][:,slice_index]

    ax.set_title("{0} (Densité) @ t = {1} s".format(name, time))
    ax.plot(dset[:], data, **kwargs)
    ax.set_ylabel("Densité $m^{-3}$")
    ax.set_xlabel('$x$')

    if plot:
        plt.show()

def plot_density(filepath):
    """Renvoie la densite stockée dans le fichier `filepath` (format HDF5)"""
    # Load file
    f = h5py.File(filepath, 'r')
    dset_x = f['x'][:]
    dset_y = f['y'][:]

    xm, ym = np.meshgrid(dset_x, dset_y)

    # Extracting info
    name = f['metadata'].attrs.get("name")
    T_end = f["metadata"].attrs.get("T end")

    plt.title("{0} (Densité) @ t = {1} s".format(name, T_end))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.pcolormesh(xm, ym, f["rho"][:])
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
    pass