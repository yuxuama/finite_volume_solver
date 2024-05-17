from utils import init_param
from solve import solve
from in_file_maker import sod_shock_tube, riemann_problem_2d, rt_instability, hydrostatic
from plot import plot_density, plot_slice
import h5py
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    params = init_param('test_RT_instability.ini.txt')
    rt_instability(params, 0.01)
    solve(params)
    
    f = h5py.File(params[-1])
    mass_a = f["rho"][:].T

    f = h5py.File(params[-2])
    mass_b= f["data"][1:params[2]+1,1:params[3]+1,0]

    diff = mass_a - mass_b
    print(np.sum(diff))

    #plt.plot(diff[25])
    #plt.show()

