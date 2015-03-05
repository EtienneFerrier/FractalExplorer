Notice d’utilisation

====================



Le code est disposé de la façon suivante :

- Une classe par implémentation de l’algorithme (Mandel.hpp, MandelGPU.cu, BigMandel.hpp, BigMandelGPU.cu).

- La classe BigFloat implémente les grands flottants et leurs opérations arithmétiques pour le CPU. CUDA ne permet pas l’utilisation de classes mais les grands flottants sont implémentés sur GPU en utilisant les mêmes structures sur CPU, directement dans BigMandelGPU.cu.

- La classe Affichage gère la fenêtre d’affichage fondée sur SDL.

- La classe Events gère l’état de l’affichage en effectuant les opération de zoom et de dezoom liées aux clics de l’utilisateur.

- Le main est implémenté dans kernel.cu, détecte le clic de l’utilisateur et appelle les fonctions Event appropriée.

- Le fichier Parametres.hpp gère les paramètres globaux du programme. C’est là qu’on décide de la résolution, de la précision des flottants, du nombre d’itérations et du choix CPU/GPU. 



L’utilisateur souhaitant simplement observer les résultats du programme peut se contenter de modifier le fichier Parametres.hpp à sa convenance.


Etienne Ferrier
Gurvan L'Hostis
https://github.com/EtienneFerrier/FractalExplorer