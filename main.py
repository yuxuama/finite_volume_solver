from utils import init_param, extract_parameter
from solve import solve
from in_file_maker import sod_shock_tube

if __name__ == '__main__':
    params = init_param('test.ini.txt')
    print(params)
    sod_shock_tube(params, direction=(1, 0))
    solve(params) # Solve and create output file

    
