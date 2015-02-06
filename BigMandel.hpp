/*
Cette classe implémente le calcul de l'ensemble de Mandelbrot sur CPU sur des flottants arbitrairement grands.
*/

#pragma once

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




class BigMandel {

public:

	// Convertit RGB en Uint32
	static Uint32 couleur(int r, int g, int b)
	{
		return (((r << 8) + g) << 8) + b;
	}

	// Methode de coloration en couleur 32bits version sombre
	// Augmenter divFactor permet d'avoir une fréquence plus importante de variation des couleurs (1 par defaut)
	static Uint32 computeColor_32_DARK(int countMax, int count, int divFactor)
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
	static Uint32 computeColor_32_CLASSIC(int countMax, int count, int divFactor)
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
	inline static int iteratePoint(BigFloat& xStart, BigFloat& yStart, int& nbIterations) {
		int count = 0;
		BigFloat *x = new BigFloat();
		BigFloat *y = new BigFloat();
		BigFloat *xx = new BigFloat();
		BigFloat *yy = new BigFloat();
		BigFloat *temp = new BigFloat();
		while (count < nbIterations && temp->base < 4.)
		{
			delete temp;
			temp = new BigFloat();
			BigFloat::mult(*x, *y, *temp);
			// x’ = xx - yy
			BigFloat::negate(*yy);
			BigFloat::add(*xx, *yy, *x);
			// y’ = 2xy
			BigFloat::add(*temp, *temp, *y);
			// x’ = x + xStart
			BigFloat::add(xStart, *x);
			// y’ = y + yStart
			BigFloat::add(yStart, *y);
			//xx = x * x
			delete xx;
			xx = new BigFloat();
			BigFloat::mult(*x, *x, *xx);
			//xx = y * y
			delete yy;
			yy = new BigFloat();
			BigFloat::mult(*y, *y, *yy);
			//temp = xx + yy
			BigFloat::add(*xx, *yy, *temp);

			count++;
			
		}
		delete x;
		delete y;
		delete xx;
		delete yy;
		delete temp;
		return count;
	}

	//inline static int iteratePoint(BigFloat& xStart, BigFloat& yStart, int& nbIterations) {
	//	int count = 0;
	//	BigFloat x, y, xx, yy, temp;
	//	while (count < nbIterations && temp.base < 4.)
	//	{
	//		temp = BigFloat();
	//		BigFloat::mult(x, y, temp);
	//		// x’ = xx - yy
	//		BigFloat::negate(yy);
	//		BigFloat::add(xx, yy, x);
	//		// y’ = 2xy
	//		BigFloat::add(temp, temp, y);
	//		// x’ = x + xStart
	//		BigFloat::add(xStart, x);
	//		// y’ = y + yStart
	//		BigFloat::add(yStart, y);
	//		//xx = x  x
	//		xx = BigFloat();
	//		BigFloat::mult(x, x, xx);
	//		//xx = y  y
	//		yy = BigFloat();
	//		BigFloat::mult(y, y, yy);
	//		//temp = xx + yy
	//		BigFloat::add(xx, yy, temp);

	//		count++;

	//	}
	//	return count;
	//}

	// Calcule la couleur d'un point de l'ensemble de Mandelbrot en fonction d'une méthode de coloration et d'un nombre d'itérations.
	static Uint32 computeColor(BigFloat& xStart, BigFloat& yStart, int methode, int nbIterations)
	{
		int count;
		//BigFloat temp;
		switch (methode)
		{
		case MANDEL_32_DARK:
			count = iteratePoint(xStart, yStart, nbIterations);
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
	static void computeMandel(Uint32* result, BigFloat& xCenter, BigFloat& yCenter, BigFloat& scale)
	{
		for (int i = 0; i < WIDTH; i++)
			for (int j = 0; j < HEIGHT; j++)
			{
				BigFloat x, y, temp;
				BigFloat::mult((-0.5f + (float)i / WIDTH), scale, x);
				BigFloat::mult((-0.5f + (float)j / HEIGHT), scale, temp);
				BigFloat::mult((HEIGHT / ((float)WIDTH)), temp, y);
				BigFloat::add(xCenter, x);
				BigFloat::add(yCenter, y);

				if (i == 30 && j == 25)
					std::cout << "Breakpoint" << std::endl;
				result[j*WIDTH + i] =
					computeColor(x,	y, MANDEL_32_DARK, NB_ITERATIONS);
			}

	}

};