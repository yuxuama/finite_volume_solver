from utils import init_param, param_struct
from solve import solve
from in_file_maker import *
import argparse

# --- MAIN FILE
# Calcul l'évolution temporelle du fluide suivant les paramètres fixés dans le fichier .ini


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Finite volume solver',
        description='A solver for fluid dynamic and thermal fluidic'
    )
    parser.add_argument('init_file', help='Nom du fichier .ini')
    args = parser.parse_args()

    params, init_func, kwargs = init_param('./init_files/' + args.init_file + '.txt')
    
    for i in range(len(param_struct)):
        print(param_struct[i][0] + '=', params[i])

    solve(params, init_func, **kwargs)

