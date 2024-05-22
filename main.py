from utils import init_param
from solve import solve
from in_file_maker import sod_shock_tube, riemann_problem_2d, rt_instability, hydrostatic, simple_convection
from plot import plot_2d, plot_slice
import h5py
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    params = init_param('test_convection.ini.txt')
    simple_convection(params, -6, 1, 10, 1e-5, 1, 0.5)
    #solve(params)

