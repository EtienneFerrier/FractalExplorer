/*
Cette classe implémente le calcul de l'ensemble de Mandelbrot sur CPU sur des flottants arbitrairement grands.
*/

#pragma once

#include "Parametres.hpp"
#if !GPU
#if BIG_FLOAT_SIZE > 0

#include <SDL2/SDL.h>
#include "Complexe.hpp"
#include <iostream>
#include "BigFloat.hpp"


// Fractale de Mandelbrot couleur 32bits version sombre
#define MANDEL_32_DARK 3
// Fractale de Mandelbrot couleur 32bits version classique
#define MANDEL_32_CLASSIC 4
// Fractale de Mandelbrot en N&B
#define MANDEL_NB 1
// Cercle en N&B (pour faire des tests)
#define CERCLE_NB 2



BigFloat *xStart = new BigFloat();
BigFloat *yStart = new BigFloat();
BigFloat *x = new BigFloat();
BigFloat *y = new BigFloat();
BigFloat *xx = new BigFloat();
BigFloat *yy = new BigFloat();
BigFloat *temp = new BigFloat();
BigFloat *temp2 = new BigFloat();

void initializePointers() {
	x = new BigFloat();
	y = new BigFloat();
	xx = new BigFloat();
	yy = new BigFloat();
	temp = new BigFloat();
	temp2 = new BigFloat();
}

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
int iteratePoint(/*BigFloat& xStart, BigFloat& yStart,*/ int& nbIterations) {
	int count = 0;
	x->reset();
	y->reset();
	xx->reset();
	yy->reset();
	temp->reset();
	while (count < nbIterations && temp->decimals[0] < 4.)
	{
		temp->reset();
		BigFloat::mult(*x, *y, *temp);
		// x’ = xx - yy
		BigFloat::negate(*yy);
		BigFloat::add(*xx, *yy, *x);
		// y’ = 2xy
		BigFloat::add(*temp, *temp, *y);
		// x’ = x + xStart
		temp2->copy(*x);
		BigFloat::add(*xStart, *temp2, *x);
		// y’ = y + yStart
		temp2->copy(*y);
		BigFloat::add(*yStart, *temp2, *y);
		//xx = x * x
		xx->reset();
		BigFloat::mult(*x, *x, *xx);
		//xx = y * y
		yy->reset();
		BigFloat::mult(*y, *y, *yy);
		//temp = xx + yy
		BigFloat::add(*xx, *yy, *temp);

		count++;
			
	}
	return count;
}


// Calcule la couleur d'un point de l'ensemble de Mandelbrot en fonction d'une méthode de coloration et d'un nombre d'itérations.
Uint32 computeColor(/*BigFloat& xStart, BigFloat& yStart,*/ int methode, int nbIterations)
{
	int count;
	switch (methode)
	{
	case MANDEL_32_DARK:
		count = iteratePoint(/*xStart, yStart,*/ nbIterations);
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
void computeBigMandel(Uint32* result, BigFloat& xCenter, BigFloat& yCenter, BigFloat& scale)
{
	for (int i = 0; i < WIDTH; i++)
		for (int j = 0; j < HEIGHT; j++)
		{
			xStart->reset();
			yStart->reset();
			temp->reset();
			BigFloat::mult((-0.5f + (float)i / WIDTH), scale, *xStart);
			BigFloat::mult((-0.5f + (float)j / HEIGHT)*(HEIGHT / ((float)WIDTH)), scale, *yStart);
			temp->copy(*xStart);
			BigFloat::add(xCenter, *temp, *xStart);
			temp->copy(*yStart);
			BigFloat::add(yCenter, *temp, *yStart);
			result[j*WIDTH + i] =
				computeColor(MANDEL_32_DARK, NB_ITERATIONS);
		}

}

#endif
#endif