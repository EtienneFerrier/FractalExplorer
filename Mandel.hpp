/*
Cette classe implémente le calcul de l'ensemble de Mandelbrot sur CPU.
L'affichage n'est pas géré.
Optimisation possible.
*/

#pragma once

#include "Parametres.hpp"
#if !GPU
#if BIG_FLOAT_SIZE == 0

#include <SDL2/SDL.h>
#include "Complexe.hpp"
#include <iostream>

// Fractale de Mandelbrot couleur 32bits version sombre
#define MANDEL_32_DARK 3
// Fractale de Mandelbrot couleur 32bits version classique
#define MANDEL_32_CLASSIC 4
// Fractale de Mandelbrot en N&B
#define MANDEL_NB 1
// Cercle en N&B (pour faire des tests)
#define CERCLE_NB 2


// Convertit RGB en Uint32
Uint32 couleur(int r, int g, int b)
{
	return (((r << 8) + g) << 8) + b;
}

// Methode de coloration en couleur 32bits version sombre
// Augmenter divFactor permet d'avoir une fréquence plus importante de variation des couleurs (1 par defaut)
Uint32 computeColor_32_DARK(int countMax, int count, int divFactor)
{
	int k = ((count * divFactor) % countMax);
	int p = countMax / 4;
	int m = k % p;

	if (k < p)
		return couleur(0, 0, 255 * m / p);
	else if (k < 2 * p)
		return couleur(255 * m / p, 0, 255);
	else if (k < 3 * p)
		return couleur(255, 0, (255 - 255 * m / p));
	else return couleur(255, 255 * m / p, 0);

}

// Methode de coloration en couleur 32bits version classique
// Augmenter divFactor permet d'avoir une fréquence plus importante de variation des couleurs (1 par defaut)
Uint32 computeColor_32_CLASSIC(int countMax, int count, int divFactor)
{
	int k = ((count * divFactor) % countMax);
	int p = countMax / 4;
	int m = k % p;

	if (k < p)
		return couleur(255, 255 * m / p, 0);
	else if (k < 2 * p)
		return couleur(255 - 255 * m / p, 255, 0);
	else if (k < 3 * p)
		return couleur(0, 255, 255 * m / p);
	else return couleur(0, 255 - 255 * m / p, 255);

}

// Itère la fonction génératrice sur un point
inline int iteratePoint(Complexe& c, int& nbIterations) {
	int count = 0;
	float x = 0;
	float y = 0;
	float xSquare = 0;
	float ySquare = 0;
	float temp;
	while (count < nbIterations && (xSquare + ySquare)  < 4.)
	{
		//z.mult(z);
		temp = x;
		x = xSquare - ySquare;
		y *= temp;
		y += y;
		x += c.x;
		y += c.y;
		count++;
		xSquare = x * x;
		ySquare = y * y;
	}
	return count;
}

// Calcule la couleur d'un point de l'ensemble de Mandelbrot en fonction d'une méthode de coloration et d'un nombre d'itérations.
Uint32 computeColor(float x, float y, int methode, int nbIterations)
{
	int count;
	Complexe c(x, y);
	switch (methode)
	{
	case MANDEL_32_DARK:
		count = iteratePoint(c, nbIterations);
		if (count == -1)
			return couleur(0, 0, 0);
		else return computeColor_32_DARK(nbIterations, count, 1);
	default:
		std::cout << "Pas de methode de coloration" << std::endl;
		return couleur(255, 0, 0);
	}
}

// Methode de test.
// Calcule l'ensemble de Mandelbrot sur le carre [-2, 2]x[-2, 2] avec une coloration N&B (10 itérations).
void computeMandel(Uint32* result, Complexe& center, float scale)
{
	for (int i = 0; i < WIDTH; i++)
		for (int j = 0; j < HEIGHT; j++)
		{
			result[j*WIDTH + i] = 
				computeColor(center.x + scale*(-0.5f + (float)i / WIDTH),
				(center.y + scale*(-0.5f + (float)j / HEIGHT))*HEIGHT / ((float)WIDTH),
				MANDEL_32_DARK,
				NB_ITERATIONS);
		}

}


#endif
#endif