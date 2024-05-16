from utils import init_param
from solve import solve
from in_file_maker import sod_shock_tube, riemann_problem_2d, rt_instability, hydrostatic
from plot import plot_density, plot_slice
import h5py
import numpy as np

if __name__ == '__main__':
    params = init_param('test_hydrostatic.ini.txt')
    #hydrostatic(params)
    solve(params)
    
    f = h5py.File(params[-1])
    mass_after = np.sum(f["rho"][:])

    f = h5py.File(params[-2])
    mass_before = np.sum(f["data"][1:params[2]+1,1:params[3]+1,0])
    print(mass_after - mass_before)
