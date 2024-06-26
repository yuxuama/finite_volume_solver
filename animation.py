import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from utils import extract_parameter, get_temp_from_pressure, get_potential_temp, get_modified_potential_temp
import h5py
import os

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

selecter = {
    "rho": ("rho", "Densité"),
    "u": ("speed x", "Vitesse selon x"),
    "v": ("speed y", "Vitesse selon y"),
    "p": ("pressure", "Pression"),
    "mx": ("momentum x", "Impulsion selon x"),
    "my": ("momentum y", "Impulsion selon y"),
}


def animate_quantity(dirpath, quantity, frames=None, rest_time=200, **kwargs):
    """Anime la `quantity` en fonction des données des fichiers contenu dans le dossier `dirpath`
    La durée du `rest_time` doit être donnée en ms
    """

    quantity, title = selecter[quantity]

    files = [dirpath + f for f in os.listdir(dirpath)]
    files.sort()
    
    if frames is None:
        frames = len(files) - 2

    params = extract_parameter(h5py.File(files[0], 'r')['metadata'])
    nx = params[p_nx]
    ny = params[p_ny]
    Lx = params[p_Lx]
    Ly = params[p_Ly]
    T_end = params[p_T_end]
    freq = 1/params[p_T_io]

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    
    x, y = np.meshgrid(x, y)

    fig, ax = plt.subplots(1, 1)
    mesh = ax.pcolormesh(x, y, h5py.File(files[2], 'r')[quantity][:], **kwargs)
    ax.set(title=f"{title} sur {T_end} s @ f_io = {freq} Hz",
           xlabel="$x$",
           ylabel="$y$"
    )
    ax.set_aspect('equal', adjustable='box')

    def animate(frame): 
        data = h5py.File(files[frame+2], 'r')[quantity][:]
        mesh.set_array(data.flatten())
        mesh.set_norm(Normalize(vmin=np.min(data), vmax=np.max(data)))
        if frame == frames-2:
            print("Animation terminée")
        return mesh,
    
    ani = FuncAnimation(fig, animate, frames=frames, interval=rest_time, blit=True, repeat=False)
    return ani

def animate_temperature(dirpath, potential=False, frames=None, rest_time=200, **kwargs):
    """Fait une animation de la température au cours du temps à partir des sauvegardes
    du dossier `dirpath`"""

    files = [dirpath + f for f in os.listdir(dirpath)]
    files.sort()

    if frames is None:
        frames = len(files) - 2

    params = extract_parameter(h5py.File(files[0], 'r')['metadata'])

    nx = params[p_nx]
    ny = params[p_ny]
    Lx = params[p_Lx]
    Ly = params[p_Ly]
    T_end = params[p_T_end]
    freq = 1/params[p_T_io]

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    
    x, y = np.meshgrid(x, y)

    if potential:
        selecter = get_potential_temp
    else:
        selecter = get_temp_from_pressure

    fig, ax = plt.subplots(1, 1)
    f1 = h5py.File(files[2], 'r')
    place_holder = selecter(f1['pressure'][:], f1['rho'][:], params)
    mesh = ax.pcolormesh(x, y, place_holder, **kwargs)
    ax.set(title=f"Temperature sur {T_end} s @ f_io = {freq} Hz",
           xlabel="$x$",
           ylabel="$y$"
    )
    ax.set_aspect('equal', adjustable='box')
    
    def animate(frame):
        f = h5py.File(files[frame+2], 'r')
        press = f['pressure'][:]
        rho =  f['rho'][:]
        temperature = selecter(press, rho, params)
        mesh.set_array(temperature.flatten())
        mesh.set_norm(Normalize(vmin=np.min(temperature), vmax=np.max(temperature)))
        if frame == frames-1:
            print("Animation terminée")
        return mesh,
    
    ani = FuncAnimation(fig, animate, frames=frames, interval=rest_time, blit=True, repeat=False)
    return ani

def animate_profile_potential_temperature(dirpath, xlim=None,frames=None, rest_time=200, **kwargs):
    """Anime le profil de température
    Pour le moment ne fait que la température potentielle modifiée
    """
    files = [dirpath + f for f in os.listdir(dirpath)]
    files.sort()

    if frames is None:
        frames = len(files) - 2

    params = extract_parameter(h5py.File(files[0], 'r')['metadata'])

    ny = params[p_ny]
    Ly = params[p_Ly]
    T_end = params[p_T_end]
    freq = 1/params[p_T_io]

    y = np.linspace(0, Ly, ny)

    fig, ax = plt.subplots(1, 1)
    f1 = h5py.File(files[2], 'r')
    place_holder = get_potential_temp(f1['pressure'][:], f1['rho'][:], params)
    line, = ax.plot(np.mean(place_holder, axis=1), y, **kwargs)
    ax.set(title=f"Temperature potentielle sur {T_end} s @ f_io = {freq} Hz",
           xlabel="Température",
           ylabel="$y$",
           adjustable="box",
    )
    if xlim is not None:
        ax.set_xlim(xlim)

    def animate(frame):
        f = h5py.File(files[frame+2], 'r')
        press = f['pressure'][:]
        rho =  f['rho'][:]
        temperature = get_potential_temp(press, rho, params)
        temperature = np.mean(temperature, axis=1)
        line.set_xdata(temperature)
        if frame == frames-1:
            print("Animation terminée")
        return line,
    
    ani = FuncAnimation(fig, animate, frames=frames, interval=rest_time, blit=True, repeat=False)
    return ani

if __name__ == '__main__':
    dir = './out/convection_confined_low_rapport/'
    #ani = animate_temperature(dir, potential=True, rest_time=100, cmap='coolwarm', shading='auto')
    #ani.save("plume devient shear temp.mp4")
    ani = animate_quantity(dir, 'v', rest_time=100, cmap='plasma', shading='auto')
    #ani = animate_profile_potential_temperature(dir)
    plt.show()