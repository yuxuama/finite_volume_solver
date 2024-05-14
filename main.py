from utils import init_param
from solve import solve
from plot import plot_density
from in_file_maker import sod_shock_tube

p_gamma = 0
p_N = 1
p_T_end = 2
p_CFL = 3
p_BC = 4
p_freq_out = 5
p_name = 6
p_in = 7
p_out = 8


if __name__ == '__main__':
    params = init_param('test.ini.txt')
    sod_shock_tube(params) # Create input file
    solve(params) # Solve and create output file
    plot_density(params[p_out])