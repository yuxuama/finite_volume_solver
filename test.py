# Fichier qui permet de tester la validité du programme

import os
import numpy as np
import h5py
from utils import *
from scipy.optimize import curve_fit

def conservation(dirpath, quantity, dS):
    """Vérifie que la quantité `quantity` est bien conservée
    `dir` est le dossier dans lequel on retrouve toutes les sauvegardes
    """
    files = [dirpath + f for f in os.listdir(dirpath)]
    files.sort()

    first_value = 0
    quantity_sum = []

    for i in range(2, len(files)): # On commence à 1 pour éviter le fichier des énergies
        data = h5py.File(files[i], 'r')[quantity][:]
        temp = np.sum(data) * dS
        if i == 1:
            first_value = temp
        else:
            quantity_sum.append(temp - first_value)
    
    return quantity_sum

def grad_shear_th(Q, params, kx, kz, start_shear):
    """Calcul le gradient de shear théorique nécessaire pour avoir une zone de stabilité"""

    Ly = params[p_Ly]
    ny = params[p_ny]
    dy = Ly / ny

    # Calcul de la température potentielle

    theta = get_potential_temp(Q[:, :, j_press], Q[:, :, j_mass], params)
    theta = np.log(np.mean(theta, axis=0))

    # On en prend le gradient

    grad_theta = np.zeros(ny)
    for i in range(1, ny+1):
        grad_theta[i-1] = 0.5 * (theta[i] - theta[i-1]) + 0.5 * (theta[i+1] - theta[i])
        grad_theta[i-1] = grad_theta[i-1] / dy
    

    grad_theta = grad_theta * params[p_g] * kx / (kz * params[p_ht])
    return np.min(grad_theta[int(Ly/start_shear):-1])

def compute_growth_rate(filepath, time_stop):
    """Calcul le taux de croissance de la perturbation"""

    f = h5py.File(filepath + 'energies.h5', 'r')
    time = f['time'][:]
    data = f['ekin y'][:]
    mask = time < time_stop
    time = time[mask]
    data = data[mask]

    model = lambda t, a, w: a * np.exp(w * t)

    popt, _ = curve_fit(model, time, data)

    return popt[1] / 2

def pression_isotherme(z, T, rho_grd, gamma, cv, g):
    """Renvoie le profil de pression en supposant que c'est isotherme"""
    P_grd = cv * (gamma - 1) * rho_grd * T
    return P_grd * np.exp(- g * z / (gamma * cv * T))

def predict_plume(T_up, T_down, rho_grd, gamma, cv, g):
    """Calcule la hauteur des plumes théorique en supposant que le profil de 
    pression ne change pas au cours du temps (ce qui est plutôt vérifié expérimentatement)
    On suppose de plus que le logarithme de la température est linéaire (ordre 1)
    Renvoie également la température potentielle stationnaire prédite par ce modèle
    """ 
    L = gamma * cv * T_up * np.log(T_down / T_up) / g
    pow = (gamma - 1) / gamma
    Tpot_th = T_up * np.power(pression_isotherme(L, T_up, rho_grd, gamma, cv, g), -pow)
    return L, Tpot_th

if __name__ == '__main__':
    """
    dirpath = "./out/layer/"
    dS = 1e-4
    mass = conservation(dirpath, 'rho', dS)
    mx = conservation(dirpath, 'momentum x', dS)
    my = conservation(dirpath, 'momentum y', dS)
    print(max(mass), max(mx), max(my))
    """
    T_down = np.linspace(1.2, 4, 200)
    #plt.plot(T_down, predict_plume(1, T_down, 1, 1.4, 1, 1)[0])
    #plt.show()    
    print(predict_plume(1, 1.6, 1, 1.4, 1, 1))
    
