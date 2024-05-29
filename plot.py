import numpy as np
import matplotlib.pyplot as plt
import h5py
from utils import extract_parameter, get_temp_from_pressure, get_potential_temp

p_gamma = 0 # Pour le tuple des paramètres
p_g = 1
p_ht = 2
p_k = 3
p_cv = 4
p_nx = 5
p_ny = 6
p_Lx = 7
p_Ly = 8
p_T_end = 9
p_CFL = 10
p_BC = 11
p_T_io = 12
p_name = 13
p_out = 14

def selecter(filename, quantity):
    """A partir de la quantité que l'on veut afficher renvoie les bonnes données à passer aux
    fonctions de plot"""

    f = h5py.File(filename, 'r')
    params = extract_parameter(f['metadata'])

    if quantity == 'rho':
        quantity = "rho"
        title = "Densité"
        ylabel = "Densité ($kg.m^{-3}$)"
        data = f[quantity][:]
    elif quantity == 'u':
        quantity = "speed x"
        title = "Vitesse en horizontale"
        ylabel = "Densité ($kg.m^{-3}$)"
        data = f[quantity][:]
    elif quantity == 'v':
        quantity = "speed y"
        title = "Vitesse verticale"
        ylabel = "Vitesse ($m.s$)"
        data = f[quantity][:]
    elif quantity == 'p':
        quantity = "pressure"
        title = "Pression"
        ylabel = "Pression ($Pa$)"
        data = f[quantity][:]
    elif quantity == 'mx':
        quantity = "momentum x"
        title = "Quantité de mouvement selon x"
        ylabel = "Impulsion x"
        data = f[quantity][:]
    elif quantity == "my":
        quantity = "momentum y"
        title = "Quantité de mouvement selon y"
        ylabel = "Impulsion y"
        data = f[quantity][:]
    elif quantity == "T":
        quantity = "Temperature"
        title = "Température"
        ylabel = "Température (K)"
        press = f["pressure"][:]
        rho = f["rho"][:]
        data = get_temp_from_pressure(press, rho, params)
    elif quantity == "Tpot":
        quantity = "Temperature"
        title = "Température potentielle"
        ylabel = "Température (K)"
        press = f["pressure"][:]
        rho = f["rho"][:]
        data = get_potential_temp(press, rho, params)
    elif quantity == "Tpots":
        quantity = "Temperature"
        title = "Température potentielle modifiée"
        ylabel = "Température (K)"
    
    return data, params, f, quantity, title, ylabel

def get_time(filename, params):
    """Pour un fichier donné renvoie le temps auquel le fichier fut pris"""
    T_end = params[p_T_end]
    T_io = params[p_T_io]
    zeros = int(np.log10(T_end / T_io))
    n = len(filename)
    number = int(filename[n-zeros-1::])
    return number * T_io

def plot_slice(filepath, quantity, slice_index, axis, ax=None, **kwargs):
    """Affiche une tranche de la densité issue des datas du fichier HDF5 `filepath` selon l'axe `axis` et pour l'indice `slice_index`
    `ax` permet de plot sur une autre figure
    Si la quantité à plot est la température affiche le profil de température
    """
    data, params, f, quantity, title, ylabel = selecter(filepath, quantity)

    # Axe des abscisse
    if axis == 0:
        axis = 'x'
        data = data[slice_index]
    elif axis == 1:
        axis = 'y'
        data = data[:,slice_index]
    space = f[axis][:]

    # Extraction d'information
    time = get_time(filepath, params)
    name = params[p_name]

    # Plot
    plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot = True

    if quantity == "Temperature":
        ax.plot(data, space, **kwargs)
    else:
        ax.plot(space, data, **kwargs)

    if plot:
        if quantity == "Temperature":
            ax.set_xlabel(ylabel)
            ax.set_ylabel(f'${axis}$')
        else:
            ax.set_ylabel(ylabel)
            ax.set_xlabel(f'${axis}$')
        ax.set_title("{0} ({1}) @ t = {2} s".format(name, title, time))
        plt.show()

def plot_hybrid_slice(filepath, a, b , quantity, ax=None, **kwargs):
    """Permet de faire les coupes même si ce n'est pas aligné avec l'un des axes
    `a` et `b` paramétrisent la droite selon laquelle on veut faire une coupe
    """
    data, params, f, quantity, title, ylabel = selecter(filepath, quantity)

    # Extraction d'information
    time = time = get_time(filepath, params)
    name = params[p_name]
    Ly = params[p_Ly]
    ny = params[p_ny]
    nx = params[p_nx]
    dy = Ly/ny

    # Plot
    plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot = True

    # Extraction des données selon la droite

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

def plot_mean_profile(filepath, quantity, axis, ax=None, **kwargs):
    """Trace le profil de la quantité `quantity` moyenne selon un axe"""
    data, params, f, quantity, title, ylabel = selecter(filepath, quantity)

    # Axe des abscisse
    if axis == 0:
        axis = 'x'
        data = np.mean(data, axis=0)
    elif axis == 1:
        axis = 'y'
        data = np.mean(data, axis=1)
    space = f[axis][:]

    # Extraction d'information
    time = get_time(filepath, params)
    name = params[p_name]

    # Plot
    plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot = True

    if quantity == "Temperature":
        ax.plot(data, space, **kwargs)
    else:
        ax.plot(space, data, **kwargs)

    if plot:
        if quantity == "Temperature":
            ax.set_xlabel(ylabel)
            ax.set_ylabel(f'${axis}$')
        else:
            ax.set_ylabel(ylabel)
            ax.set_xlabel(f'${axis}$')
        ax.set_title("{0} ({1}) @ t = {2} s".format(name, title, time))
        plt.show()

def plot_2d(filepath, quantity, ax=None, **kwargs):
    """Plot la densite stockée dans le fichier `filepath` (format HDF5)"""
    # Load file

    data, params, f, quantity, title, _ = selecter(filepath, quantity)

    dset_x = f['x'][:]
    dset_y = f['y'][:]

    xm, ym = np.meshgrid(dset_x, dset_y)

    # Extracting info
    name = params[p_name]
    time = get_time(filepath, params)

    # Plot
    plot = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plot = True

    ax.set_aspect('equal', adjustable='box')
    ax.pcolormesh(xm, ym, data, **kwargs)

    if plot:
        ax.set_title("{0} ({1}) @ t = {2} s".format(name, title, time))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')        
        plt.show()

def plot_energy(filepath):
    """Trace les diagrammes d'énergie"""
    f = h5py.File(filepath, 'r')
    params = extract_parameter(f['metadata'])

    time = f['time'][:]
    ekin_x = f['ekin x'][:]
    ekin_y = f['ekin y'][:]

    name = f['metadata'].attrs.get("name")
    time = get_time(filepath, params)

    fig, ax = plt.subplots(1, 1)
    fig.suptitle("{0} | évolution sur {1} s".format(name, time))
    
    ax.semilogy(time, ekin_x, '--b', label="selon x")
    ax.semilogy(time, ekin_y, 'b', label="selon y")
    ax.set(title="Evolution de l'énergie cinétique", xlabel="$t$ (s)", ylabel="Energie (J)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    file = './out/layer/'
    plot_mean_profile(file + 'save_30', 'Tpot', 1)

    