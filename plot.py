import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils import extract_parameter, get_temp_from_pressure

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

def selecter(quantity):
    """À partir de la quantité que l'on veut afficher détermine les bons labels et titres"""
    if quantity == 'rho':
        quantity = "rho"
        title = "Densité"
        ylabel = "Densité ($kg.m^{-3}$)"
    elif quantity == 'u':
        quantity = "speed x"
        title = "Vitesse en horizontale"
        ylabel = "Densité ($kg.m^{-3}$)"
    elif quantity == 'v':
        quantity = "speed y"
        title = "Vitesse verticale"
        ylabel = "Vitesse ($m.s$)"
    elif quantity == 'p':
        quantity = "pressure"
        title = "Pression"
        ylabel = "Pression ($Pa$)"
    elif quantity == 'mx':
        quantity = "momentum x"
        title = "Quantité de mouvement selon x"
        ylabel = "Impulsion x"
    elif quantity == "my":
        quantity = "momentum y"
        title = "Quantité de mouvement selon y"
        ylabel = "Impulsion y"
    
    return quantity, title, ylabel

def plot_slice(filepath, quantity, slice_index, axis, ax=None, **kwargs):
    """Affiche une tranche de la densité issue des datas du fichier HDF5 `filepath` selon l'axe `axis` et pour l'indice `slice_index`
    `ax` permet de plot sur une autre figure
    """

    if axis == 0:
        axis = 'x'
    elif axis == 1:
        axis = 'y'

    quantity, title, ylabel = selecter(quantity)

    # Load file
    f = h5py.File(filepath, 'r')
    meta = f['metadata']
    dset = f[axis]

    # Extraction d'information
    time = meta.attrs.get("T end")
    name = meta.attrs.get("name")

    # Getting data   
    if axis == 'x':
        data = f[quantity][slice_index]
    elif axis == 'y':
        data = f[quantity][:,slice_index]

    # Plot
    plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot = True

    ax.plot(dset[:], data, **kwargs)

    if plot:
        ax.set_title("{0} ({1}) @ t = {2} s".format(name, title, time))
        ax.set_ylabel(ylabel)
        ax.set_xlabel(f'${axis}$')
        plt.show()

def plot_hybrid_slice(filepath, a, b , quantity, ax=None, **kwargs):
    """Permet de faire les coupes même si ce n'est pas aligné avec l'un des axes
    `a` et `b` paramétrisent la droite selon laquelle on veut faire une coupe
    """
    quantity, title, ylabel = selecter(quantity)
    
    # Load file
    f = h5py.File(filepath, 'r')
    meta = f['metadata']

    # Extraction d'information
    time = meta.attrs.get("T end")
    name = meta.attrs.get("name")
    Ly = meta.attrs.get('Ly')
    ny = meta.attrs.get('ny')
    nx = meta.attrs.get('nx')
    dy = Ly/ny

    # Plot
    plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot = True

    # Extraction des données selon la droite

    data = f[quantity][:]
    x = f['x'][:]
    y_line = a * x + b 

    pre_mask = np.logical_and(y_line > 0, y_line < Ly)
    y_line = y_line[pre_mask]
    x = x[pre_mask]
    y = y_line

    # Conversion en indice

    y_line = y_line / dy
    y_line = y_line.astype(int)
    x_line = np.arange(0, nx)
    x_line = x_line[pre_mask]

    data = data[x_line, y_line]
    ax.plot(np.sqrt(x**2 + y**2), data, **kwargs)

    if plot:
        ax.set_title("{0} ({1}) @ t = {2} s".format(name, title, time))
        ax.set_ylabel(ylabel)
        plt.show()

def plot_2d(filepath, quantity, ax=None, **kwargs):
    """Plot la densite stockée dans le fichier `filepath` (format HDF5)"""
    # Load file

    quantity, title, _ = selecter(quantity)

    f = h5py.File(filepath, 'r')
    dset_x = f['x'][:]
    dset_y = f['y'][:]

    xm, ym = np.meshgrid(dset_x, dset_y)

    # Extracting info
    name = f['metadata'].attrs.get("name")
    T_end = f["metadata"].attrs.get("T end")

    # Plot
    plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot = True

    ax.set_aspect('equal', adjustable='box')
    ax.pcolormesh(xm, ym, f[quantity][:], **kwargs)

    if plot:
        ax.set_title("{0} ({1}) @ t = {2} s".format(name, title, T_end))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')        
        plt.show()

def plot_temperature(filepath, ax=None, **kwargs):
    """Plot la température issue des données stockées dans le fichier `filepath` (format HDF5)"""
    # Load file
    f = h5py.File(filepath, 'r')
    dset_x = f['x'][:]
    dset_y = f['y'][:]

    xm, ym = np.meshgrid(dset_x, dset_y)

    # Extracting info
    name = f['metadata'].attrs.get("name")
    T_end = f["metadata"].attrs.get("T end")
    params = extract_parameter(f['metadata'])

    # Extracting temperature
    nx = dset_x.size
    ny = dset_y.size
    temp = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            temp[i, j] = get_temp_from_pressure(f['pressure'][i, j], f['rho'][i, j], params)

    # Plot
    plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot = True

    ax.set_aspect('equal', adjustable='box')
    ax.pcolormesh(xm, ym, temp, cmap='hot', **kwargs)

    if plot:
        ax.set_title("{0} (Temperature) @ t = {1} s".format(name, T_end))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')        
        plt.show()


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    file = './out/simple_convection/convection simple_18.h5'

    plot_2d(file, "rho", ax=ax[0])
    plot_2d(file, "v", ax=ax[1])
    plot_temperature(file, ax=ax[2])

    plt.show()
    