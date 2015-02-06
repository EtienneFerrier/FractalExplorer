/*
Ce fichier permet de modifier les parametres programme.
Il est normalement inclus dans tous les fichiers du projet
*/

#pragma once

#define WIDTH 300			// Largeur de la fenetre
#define HEIGHT 300				// Hauteur de la fenetre
#define ZOOM_FACTOR 0.5f			// Facteur de zoom (par clic)
#define DEZOOM_FACTOR 0.5f		// Facteur de dezoom (typiquement identique au facteur de zoom)
#define NB_ITERATIONS 50		// Nombre d’itérations maximal
#define GPU 0					// 1 if using GPU, 0 if using CPU
#define BIG_FLOAT_SIZE 4		// LE nombre de bits sur lesquels on code un grand flottant, 0 pour utiliser les floats normaux