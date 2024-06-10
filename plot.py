import numpy as np
import matplotlib.pyplot as plt
import h5py
from test import predict_plume
from utils import extract_parameter, get_temp_from_pressure, get_potential_temp, get_modified_potential_temp, compute_omega

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

def selecter(filename, quantity, pot_settings):
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
        ylabel = "Vitesse ($m.s$)"
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
        press = f["pressure"][:]
        rho = f["rho"][:]
        speed_x = f["speed x"][:]
        data = get_modified_potential_temp(press, rho, speed_x, params, *pot_settings)
    elif quantity == "logT":
        quantity = "Temperature"
        title = "log de la température"
        ylabel = "Température"
        press = f["pressure"][:]
        rho = f["rho"][:]
        data = np.log(get_temp_from_pressure(press, rho, params))
    elif quantity == "logTpot":
        quantity = "Temperature"
        title = "Log de la température potentielle"
        ylabel = "Température"
        press = f["pressure"][:]
        rho = f["rho"][:]
        data = np.log(get_potential_temp(press, rho, params))
    elif quantity == "gradadlogp":
        quantity = "pressure"
        title = "Pression"
        ylabel = "Pression ($Pa$)"
        data = np.log(f[quantity][:])
        quantity = "Temperature"
        data = -(params[p_gamma] - 1) * data / params[p_gamma]

    return data, params, f, quantity, title, ylabel

def get_time(filename, params):
    """Pour un fichier donné renvoie le temps auquel le fichier fut pris"""
    T_end = params[p_T_end]
    T_io = params[p_T_io]
    zeros = int(np.log10(T_end / T_io))
    n = len(filename)
    number = int(filename[n-zeros-1::])
    return number * T_io

def plot_slice(filepath, quantity, slice_index, axis, pot_settings=None, ax=None, **kwargs):
    """Affiche une tranche de la densité issue des datas du fichier HDF5 `filepath` selon l'axe `axis` et pour l'indice `slice_index`
    `ax` permet de plot sur une autre figure
    Si la quantité à plot est la température affiche le profil de température
    """
    data, params, f, quantity, title, ylabel = selecter(filepath, quantity, pot_settings)

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

def plot_hybrid_slice(filepath, a, b , quantity, pot_setting=None, ax=None, **kwargs):
    """Permet de faire les coupes même si ce n'est pas aligné avec l'un des axes
    `a` et `b` paramétrisent la droite selon laquelle on veut faire une coupe
    """
    data, params, f, quantity, title, ylabel = selecter(filepath, quantity, pot_setting)

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

def plot_mean_profile(filepath, quantity, axis, pot_setting=None, ax=None, **kwargs):
    """Trace le profil de la quantité `quantity` moyenne selon un axe"""
    data, params, f, quantity, title, ylabel = selecter(filepath, quantity, pot_setting)

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

def plot_2d(filepath, quantity, pot_setting=None, ax=None, **kwargs):
    """Plot la densite stockée dans le fichier `filepath` (format HDF5)"""
    # Load file

    data, params, f, quantity, title, _ = selecter(filepath, quantity, pot_setting)

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

def plot_energy(filepath, diff=bool):
    """Trace les diagrammes d'énergie
    Si `diff` est `True` alors trace l'énergie moins l'énergie initiale en valeur absolue
    """
    f = h5py.File(filepath, 'r')
    params = extract_parameter(f['metadata'])

    time_axis = f['time'][:]
    ekin_x = f['ekin x'][:]
    ekin_y = f['ekin y'][:]

    name = params[p_name]
    time = params[p_T_end]

    fig, ax = plt.subplots(1, 1)
    fig.suptitle("{0} | évolution sur {1} s".format(name, time))
    
    if diff:
        ekin_x = np.abs(ekin_x - ekin_x[0])
        ekin_y = np.abs(ekin_y - ekin_y[0])

    ax.semilogy(time_axis, ekin_x, '--b', label="selon x")
    ax.semilogy(time_axis, ekin_y, 'b', label="selon y")
    ax.set(title="Evolution de l'énergie cinétique", xlabel="$t$ (s)", ylabel="Energie (J)")
    ax.legend()
    plt.show()

def plot_omega(filepath, kx, kz):
    """Trace le profil du taux de montée"""

    f = h5py.File(filepath, 'r')
    params = extract_parameter(f['metadata'])
    Ly = params[p_Ly]
    ny = params[p_ny]

    u = f['speed x'][:]
    pressure = f['pressure'][:]
    rho = f['rho'][:]
    
    omega = np.zeros(ny)

    for i in range(ny):
        omega[i] = compute_omega(u, pressure, rho, i, kx, kz, params)
    
    # plot
    z = np.linspace(0, Ly, ny)

    plt.plot(omega, z)
    plt.xlabel('$\omega$')
    plt.ylabel('$y$')
    plt.title("Profil du taux de croissance moyen de l'instabilité")
    plt.show()

def plot_plumes(file, T_up, T_down, rho_grd, gamma, cv, g, ax=None):
    """Trace le graphique comparant le modèle de hauteur de plume avec les simulations"""
    plot = False
    if ax is None:
        plot = True
        ax = plt.subplot(1, 1, 1)
    Lth, Tpot_th = predict_plume(T_up, T_down, rho_grd, gamma, cv, g)
    print("Température potentielle stationnaire prédite:", Tpot_th)
    ax.hlines(Lth, 0, 1, 'r', linestyles='dashed', label="Hauteur donné par le modèle")
    plot_mean_profile(file, 'logTpot', 1, ax=ax, label="$\log\\theta$")
    plot_mean_profile(file, 'logT', 1, ax=ax, label="$\log T$")
    plot_mean_profile(file, 'gradadlogp', 1, ax=ax, label="$- \\nabla_{ad} \log p$")
    
    if plot:
        ax.set(ylabel="$y$", title="Test du modèle pour $T_{grd} = "+ str(T_down) + "$")
        ax.scatter(np.log(T_down), 0, s=10, c='r', label="thermostat")
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    file = './out/convection_limited/'
    plot_energy(file+'energies.h5', diff=True)
    #plot_omega(file + 'save_10', 1, 0.25)
    #plot_mean_profile(file + "save_400", 'T', 1, pot_setting=None)
    ax = plt.subplot(1, 1, 1)
    plot_plumes(file + "save_400", 1, 1.6, 1, 1.4, 1, 1, ax)
    ax.hlines(0.4, 0, 1, 'r', linestyles='dashdot', label="Début du shear")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    plt.show()
    