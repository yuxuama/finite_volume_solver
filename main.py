from utils import init_param, param_struct
from solve import solve
from in_file_maker import *
import argparse

# --- MAIN FILE
# Calcul l'évolution temporelle du fluide suivant les paramètres fixés dans le fichier .ini


if __name__ == '__main__':
    
    # Parse tous les paramètres
    parser = argparse.ArgumentParser(
        prog='Finite volume solver',
        description='A solver for fluid dynamic and thermal fluidic'
    )
    parser.add_argument('init_file', help='Nom du fichier .ini ou dossier qui contient la simulation')
    parser.add_argument('-t', '--time', type=float, help="Si pas None enclenche le mode 'resume' et store le temps supplémentaire que l'on veut simuler")
    args = parser.parse_args()

    if args.time is None:
        params, init_func, kwargs = init_param('./init_files/' + args.init_file + '.txt')
        
        # Affiche les paramètres utilisés pour la simulation
        for i in range(len(param_struct)):
            print(param_struct[i][0] + '=', params[i])
    else:
        params = None
        init_func = "resume simulation"
        kwargs = {
            "dirpath": "./out/" + args.init_file,
            "extra_time": args.time
        }
    # Résout avec les paramètres donnés le problème
    solve(params, init_func, **kwargs)

