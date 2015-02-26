/*
Ce fichier permet de modifier les parametres programme.
Il est normalement inclus dans tous les fichiers du projet
*/

#pragma once

#define WIDTH 512				// Largeur de la fenetre
#define HEIGHT 512				// Hauteur de la fenetre
#define ZOOM_FACTOR 0.5f		// Facteur de zoom (par clic)
#define DEZOOM_FACTOR 0.95f		// Facteur de dezoom (typiquement identique au facteur de zoom)
#define NB_ITERATIONS 100		// Nombre d’itérations maximal
#define GPU 1					// 1 if using GPU, 0 if using CPU
#define BIG_FLOAT_SIZE 6		// Le nombre de bits sur lesquels on code un grand flottant, 0 pour utiliser les floats normaux
#define INTERACTIVE 1			// Booléen décidant si le zoom est fixé ou interactif
#define BLOCK_X 32				// Taille des block CUDA en X
#define BLOCK_Y 1				// Taille des block CUDA en Y