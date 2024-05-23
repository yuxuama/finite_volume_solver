# Fichier qui permet de tester la validité du programme

import os
import numpy as np
import h5py

def conservation(dirpath, quantity, dS):
    """Vérifie que la quantité `quantity` est bien conservée
    `dir` est le dossier dans lequel on retrouve toutes les sauvegardes
    """
    files = [dirpath + f for f in os.listdir(dirpath)]
    files.sort()

    first_value = 0
    quantity_sum = []

    for i in range(1, len(files)): # On commence à 1 pour éviter le fichier des énergies
        data = h5py.File(files[i], 'r')[quantity][:]
        temp = np.sum(data) * dS
        if i == 1:
            first_value = temp
        else:
            quantity_sum.append(temp - first_value)
    
    return quantity_sum

if __name__ == '__main__':
    dirpath = "./out/forced_convection/"
    dS = 1e-4
    mass = conservation(dirpath, 'rho', dS)
    mx = conservation(dirpath, 'momentum x', dS)
    my = conservation(dirpath, 'momentum y', dS)
    print(max(mass), max(mx), max(my))
