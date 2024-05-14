# finite_volume_solver
Simulation de fluide à 1 ou 2 dimensions en utilisant l'algorithme des volumes finis

# Exemple d'utilisation

Voir `main.py`

Le solveur volume fini prend en argument les paramètres listé dans le fichier txt. Pour que cela fonctionne il faut un fichier d'entrée avec les conditions initiales qui peut être créer par exemple avec `sod_shock_tube`.

Le fichier `hdf5` de sortie est lu avec `plot_density`