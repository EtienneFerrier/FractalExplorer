Notice d’utilisation
====================

ATTENTION : ce code utilise la librairie SDL2 et CUDA.

Le code est disposé de la façon suivante :
- Un fichier par version de l’algorithme : 
    Mandel.hpp (CPU simple)
    MandelGPU.cu (GPU simple)
    BigMandel.hpp (CPU precis)
    BigMandelGPU.cu (GPU precis)
    
- La classe BigFloat (BigFloat.hpp) implémente les grands flottants et leurs opérations arithmétiques pour le CPU. CUDA ne permet pas l’utilisation de classes mais les grands flottants sont implémentés sur GPU en utilisant les mêmes structures que sur CPU, directement dans BigMandelGPU.cu.

- La classe Affichage (Affichage.hpp) gère la fenêtre d’affichage fondée sur SDL2.

- La classe Events (Events.hpp) gère l’état de l’affichage en effectuant les opération de zoom et de dezoom liées aux clics de l’utilisateur.

- Le main est implémenté dans kernel.cu, détecte le clic de l’utilisateur et appelle les fonctions Events appropriées.

- Le fichier Parametres.hpp gère les paramètres globaux du programme. C’est là qu’on décide de la résolution, de la précision des flottants, du nombre d’itérations et du choix CPU/GPU.

L’utilisateur souhaitant simplement observer les résultats du programme peut se contenter de modifier le fichier Parametres.hpp à sa convenance.
