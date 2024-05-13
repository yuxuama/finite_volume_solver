from utils import init_param
from solve import solve
from plot import plot_density


if __name__ == '__main__':
    params = init_param('test.ini.txt')
    solve(params)
    plot_density(params)