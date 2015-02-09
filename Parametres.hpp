/*
Ce fichier permet de modifier les parametres programme.
Il est normalement inclus dans tous les fichiers du projet
*/

#pragma once

#define WIDTH 200				// Largeur de la fenetre
#define HEIGHT 200				// Hauteur de la fenetre
#define ZOOM_FACTOR 0.5f		// Facteur de zoom (par clic)
#define DEZOOM_FACTOR 0.95f		// Facteur de dezoom (typiquement identique au facteur de zoom)
#define NB_ITERATIONS 80		// Nombre d’itérations maximal
#define GPU 0					// 1 if using GPU, 0 if using CPU
#define BIG_FLOAT_SIZE 8		// Le nombre de bits sur lesquels on code un grand flottant, 0 pour utiliser les floats normaux
#define INTERACTIVE 0			// Booléen décidant si le zoom est fixé ou interactif