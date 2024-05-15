from utils import init_param, extract_parameter
from solve import solve
from in_file_maker import sod_shock_tube
from plot import plot_density
import time

if __name__ == '__main__':
    params = init_param('test.ini.txt')
    sod_shock_tube(params, direction=(1, 1))
    solve(params)
    plot_density(params[-1])
    

    
