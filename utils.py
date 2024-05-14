import h5py

p_gamma = 0
p_N = 1
p_T_end = 2
p_CFL = 3
p_BC = 4
p_freq_out = 5
p_name = 6
p_in = 7
p_out = 8

param_struct = [  ("Gamma", float),
                ("N", int),
                ("T end", float),
                ("CFL", float),
                ("BC", str),
                ("freq out", float),
                ("name", str),
                ("input name", str),
                ("output name", str)
    ]

def init_param(filename):
    """Définit tous les paramètres de la simulation étant donné un fichier txt du bon format"""
    params = [0 for i in range(9)]
    with open(filename, 'r') as f:
        i = 0
        for line in f.readlines():

            header_size = len(param_struct[i][0]) + 1 # On inclut les :

            if param_struct[i][1] is float:
                params[i] = float(line[header_size::])
            elif param_struct[i][1] is str:
                params[i] = line[header_size::].lower().strip()
            elif param_struct[i][1] is int:
                params[i] = int(line[header_size::])
            i+=1
        
    assert params[p_BC] in ['neumann', 'periodic']
    params[p_in] = "./in/" + params[p_in]
    params[p_out] = "./out/" + params[p_out]
    return tuple(params) # Pour que numba fonctionne (objet non modifiable)

def create_all_attribute(hdf_dset, params):
    """Met en attribut d'un hdf5 tous les paramètres de la simumation"""
    for i in range(7):
        hdf_dset.attrs.create(param_struct[i][0], params[i])

# Test

if __name__ == '__main__':
    print(init_param('./test.ini.txt'))