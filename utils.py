import h5py

p_gamma = 0
p_N = 1
p_T_end = 2
p_CFL = 3
p_BC = 4
p_freq_out = 5
p_in = 6
p_out = 7

def init_param(filename):
    """Définit tous les paramètres de la simulation étant donné un fichier txt du bon format"""
    params = [0 for i in range(8)]
    struct = [  ("Gamma:", float),
                ("N:", int),
                ("T end:", float),
                ("CFL:", float),
                ("BC:", str),
                ("freq out:", float),
                ("input name:", str),
                ("output name:", str)
    ]
    with open(filename, 'r') as f:
        i = 0
        for line in f.readlines():
            if struct[i][1] is float:
                params[i] = float(line[len(struct[i][0])::])
            elif struct[i][1] is str:
                params[i] = line[len(struct[i][0])::].lower().strip()
            elif struct[i][1] is int:
                params[i] = int(line[len(struct[i][0])::])
            i+=1
        
    assert params[p_BC] in ['neumann', 'periodic']
    params[p_in] = "./in/" + params[p_in]
    params[p_out] = "./out/" + params[p_out]
    return tuple(params)

def create_all_attribute(hdf_file):
    """Met en attribut d'un hdf5 tous les paramètres de la simumation"""
    # TODO
    pass
# Test

if __name__ == '__main__':
    print(init_param('./test.ini.txt'))