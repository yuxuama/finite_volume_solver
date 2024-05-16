from utils import init_param
from solve import solve
from in_file_maker import sod_shock_tube, riemann_problem_2d
from plot import plot_density

if __name__ == '__main__':
    params = init_param('test_riemann.ini.txt')
    riemann_problem_2d(params)
    solve(params)
    plot_density(params[-1])
    

    
