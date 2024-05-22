import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from plot import selecter
from utils import extract_parameter
import h5py
import os

p_gamma = 0 # Pour les tableaux de grandeurs primitives
p_g = 1
p_cv = 2
p_nx = 3
p_ny = 4
p_Lx = 5
p_Ly = 6
p_T_end = 7
p_CFL = 8
p_BC = 9
p_freq_out = 10
p_name = 11
p_in = 12
p_out = 13

def animate_quantity(dirpath, quantity, frames, rest_time=200):
    """Anime la `quantity` en fonction des données des fichiers contenu dans le dossier `dirpath`
    La durée du `rest_time` doit être donnée en ms
    """

    quantity, title, _ = selecter(quantity)

    files = [dirpath + f for f in os.listdir(dirpath)]

    params = extract_parameter(h5py.File(files[0], 'r')['metadata'])

    nx = params[p_nx]
    ny = params[p_ny]
    Lx = params[p_Lx]
    Ly = params[p_Ly]

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    
    x, y = np.meshgrid(x, y)

    fig, ax = plt.subplots(1, 1)
    mesh = ax.pcolormesh(x, y, h5py.File(files[0], 'r')[quantity][:])

    def animate(frame): 
        data = h5py.File(files[frame], 'r')[quantity][:]
        print(files[frame])
        mesh.set_array(data)
        return mesh,
    
    ani = FuncAnimation(fig, animate, frames=frames, interval=rest_time, blit=True, repeat=False)
    return ani

if __name__ == '__main__':
    ani = animate_quantity('./out/simple_convection/', "rho", 40)
    plt.show()