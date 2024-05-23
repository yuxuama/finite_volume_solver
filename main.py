from utils import init_param
from solve import solve
from in_file_maker import sod_shock_tube, riemann_problem_2d, rt_instability, hydrostatic, simple_convection
from plot import plot_2d, plot_slice
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    params = init_param('test_convection.ini.txt')
    #simple_convection(params, -6, 1, 10, 1e-5, 1, 0.5)
    #solve(params)

    # Vérification de la conservation

    files = [params[-1] + f for f in os.listdir(params[-1])]
    files.sort()

    mass = []
    mx = []
    my = []
    for f in files:
        data = h5py.File(f, 'r')
        mass.append(
            np.sum(data['rho'][:])
        )
        mx.append(
            np.sum(data['momentum x'][:])
        )
        my.append(
            np.sum(data['momentum y'][:])
        )
    
    print(mass[0], mass[3])

    for i in range(1, len(files)):
        mass[i] = mass[i] - mass[0]
        mx[i] = mx[i] - mx[0]
        my[i] = my[i] - my[0]
    
    mass[0] = 0
    mx[0] = 0
    my[0] = 0
    print(max(mass), max(mx), max(my))


