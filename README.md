# finite_volume_solver
Simulation de fluide à 1 ou 2 dimensions en utilisant l'algorithme des volumes finis. Prend en compte les effets thermo-hydrodynamique tel que la diffusion.

# Utilisation

## Lancement d'une simulation

Afin de lancer une simulation il faut au préalable remplir une fichier d'initialisation au format `.txt` dans le dossier `init_files` (voir ce dossier pour comprendre les paramètres à rentrer)

Si tout est correctement rentré dans ce fichier alors il suffit de rentrer dans l'invite de commande: `python main.py <init_file_name>`.
*Attention:* ne pas inclure `./init_files/` au début du nom.

La simulation est alors lancée et une barre de progression devrait apparaître.

## Continuer une simulation
Si on veut rallonger le temps sur lequel on fait la simulation on peut faire la commande suivante: `python main.py -t <temps supplémentaire> <dossier contenant la simulation>`
*Attention:* ne pas oublier le slash à la fin pour le nom du dossier, ne pas inclure `./out/` au début.

## Visualisation

Voir les fichiers `plot.py` pour l'affichage de frame, et `animation.py` pour les animations au cours du temps.
*Note:* Comme chaque pas de temps est un fichier sauvegardé on peut tout à fait faire de la visualisation alors même que la simulation est toujours en cours
