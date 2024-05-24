import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from plot import selecter
from utils import extract_parameter
import h5py
import os

p_gamma = 0 # Pour les tableaux de grandeurs primitives
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
p_freq_out = 12
p_name = 13
p_in = 14
p_out = 15

def get_normalize_cmap(file_list, quantity, ratio):
    """Renvoie une normalisation globale pour la colormap
    `file_list` liste des adresses des fichiers qui forment l'animation
    `quantity` la quantité que l'on étudie
    """
    maxi = 0
    j = -1
    for i in range(len(file_list)):
        data = h5py.File(file_list[i], 'r')[quantity][:]
        temp = np.max(np.abs(data))
        if temp > maxi:
            maxi = temp
            j = i

    return Normalize(-maxi*ratio, maxi*ratio)


def animate_quantity(dirpath, quantity, frames, rest_time=200, global_norm=False, ratio=1, **kwargs):
    """Anime la `quantity` en fonction des données des fichiers contenu dans le dossier `dirpath`
    La durée du `rest_time` doit être donnée en ms
    """

    quantity, title, _ = selecter(quantity)

    files = [dirpath + f for f in os.listdir(dirpath)]
    files.sort()

    params = extract_parameter(h5py.File(files[0], 'r')['metadata'])

    nx = params[p_nx]
    ny = params[p_ny]
    Lx = params[p_Lx]
    Ly = params[p_Ly]
    T_end = params[p_T_end]
    freq = 1/params[p_freq_out]

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    
    x, y = np.meshgrid(x, y)

    norm = None
    if global_norm:
        norm = get_normalize_cmap(files, quantity, ratio)

    fig, ax = plt.subplots(1, 1)
    mesh = ax.pcolormesh(x, y, h5py.File(files[1], 'r')[quantity][:], norm=norm, **kwargs)
    ax.set(title=f"{title} sur {T_end} s @ f_io = {freq} Hz",
           xlabel="$x$",
           ylabel="$y$"
    )

    def animate(frame): 
        data = h5py.File(files[frame+1], 'r')[quantity][:]
        mesh.set_array(data)
        return mesh,
    
    ani = FuncAnimation(fig, animate, frames=frames, interval=rest_time, blit=True, repeat=False)
    return ani

if __name__ == '__main__':
    ani = animate_quantity('./out/simple_diffusion/', "rho", 10, cmap='coolwarm', shading='auto')
    plt.show()