import numpy as np
from utils import get_speed, get_pressure, extract_parameter
import h5py
import matplotlib.pyplot as plt

# Pour les paramètres N = 100, L=1, CSL=0.9, Tend=0.2

# Load file
f = h5py.File('./out/sod shock.h5', 'r')
dset = f['data']

f_ana = h5py.File('./out/sod analytique.h5', 'r')
dset = f['data']

# Extraction d'information

params = extract_parameter(dset)

N = dset.attrs.get("N")
time = dset.attrs.get("T end")
name = dset.attrs.get("name")
# Axe des abscisse
x = np.linspace(0, 1, N)

# Axe des ordonnées

mask = np.arange(1, N+1)
speed = get_speed(dset[:], mask)
pressure = get_pressure(dset[:], mask, params)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("{0} @ t = {1} s".format(name, time))

ax[0].plot(x, dset[1:N+1, 0], label="calculé")
ax[0].plot(x, f_ana['rho'][:], '--r', label="analytique")
ax[0].set(xlabel="$x$", ylabel="Densité", title="Densité")
ax[0].legend()

ax[1].plot(x, speed, label="calculé")
ax[1].plot(x, f_ana['u'][:], '--r', label="analytique")
ax[1].set(xlabel="$x$", ylabel="Vitesse", title="Vitesse")
ax[1].legend()

ax[2].plot(x, pressure, label="calculé")
ax[2].plot(x, f_ana['p'][:], '--r', label="analytique")
ax[2].set(xlabel="$x$", ylabel="Pression", title="Pression")
ax[2].legend()

plt.show()