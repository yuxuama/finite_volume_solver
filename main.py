from utils import init_param, get_speed, get_pressure
from solve import solve
from plot import plot_density, plot_all_primitive
from in_file_maker import sod_shock_tube, two_rarefaction
import numpy as np

if __name__ == '__main__':
    params = init_param('test.ini.txt')
    sod_shock_tube(params)
    solve(params) # Solve and create output file

    
