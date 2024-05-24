from utils import init_param
from solve import solve
from in_file_maker import *

if __name__ == '__main__':
    params = init_param('test_diffusion.ini.txt')
    #simple_convection(params, -6, 1, 10, 1e-5, 1, 0.5)
    simple_diffusion(params, 10, 3, 0.1, 6)
    solve(params)

