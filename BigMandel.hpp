/*
Cette classe implémente le calcul de l'ensemble de Mandelbrot sur CPU sur des flottants arbitrairement grands.
*/

#pragma once

#include <SDL2/SDL.h>
#include "Complexe.hpp"
#include <iostream>
#include "BigFloat2.hpp"

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
	static BigFloat2 *xStart;
	static BigFloat2 *yStart;
	static BigFloat2 *x;
	static BigFloat2 *y;
	static BigFloat2 *xx;
	static BigFloat2 *yy;
	static BigFloat2 *temp;
	static BigFloat2 *temp2;

	static void initializePointers() {
		x = new BigFloat2();
		y = new BigFloat2();
		xx = new BigFloat2();
		yy = new BigFloat2();
		temp = new BigFloat2();
		temp2 = new BigFloat2();
	}

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
	static int iteratePoint(/*BigFloat2& xStart, BigFloat2& yStart,*/ int& nbIterations) {
		int count = 0;
		x->reset();
		y->reset();
		xx->reset();
		yy->reset();
		temp->reset();
		while (count < nbIterations && temp->decimals[0] < 4.)
		{
			temp->reset();
			BigFloat2::mult(*x, *y, *temp);
			// x’ = xx - yy
			BigFloat2::negate(*yy);
			BigFloat2::add(*xx, *yy, *x);
			// y’ = 2xy
			BigFloat2::add(*temp, *temp, *y);
			// x’ = x + xStart
			temp2->copy(*x);
			BigFloat2::add(*xStart, *temp2, *x);
			// y’ = y + yStart
			temp2->copy(*y);
			BigFloat2::add(*yStart, *temp2, *y);
			//xx = x * x
			xx->reset();
			BigFloat2::mult(*x, *x, *xx);
			//xx = y * y
			yy->reset();
			BigFloat2::mult(*y, *y, *yy);
			//temp = xx + yy
			BigFloat2::add(*xx, *yy, *temp);

			count++;
			
		}
		return count;
	}


	// Calcule la couleur d'un point de l'ensemble de Mandelbrot en fonction d'une méthode de coloration et d'un nombre d'itérations.
	static Uint32 computeColor(/*BigFloat2& xStart, BigFloat2& yStart,*/ int methode, int nbIterations)
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
	static void computeMandel(Uint32* result, BigFloat2& xCenter, BigFloat2& yCenter, /*BigFloat2& xStart, BigFloat2& yStart,*/ BigFloat2& scale)
	{
		for (int i = 0; i < WIDTH; i++)
			for (int j = 0; j < HEIGHT; j++)
			{
				xStart->reset();
				yStart->reset();
				temp->reset();
				BigFloat2::mult((-0.5f + (float)i / WIDTH), scale, *xStart);
				BigFloat2::mult((-0.5f + (float)j / HEIGHT)*(HEIGHT / ((float)WIDTH)), scale, *yStart);
				//BigFloat2::mult(), *temp, *yStart);
				temp->copy(*xStart);
				BigFloat2::add(xCenter, *temp, *xStart);
				temp->copy(*yStart);
				BigFloat2::add(yCenter, *temp, *yStart);
				if (i == WIDTH/2 + 10 && j == HEIGHT/2+10)
					std::cout << "Break" << std::endl;
				result[j*WIDTH + i] =
					computeColor(/*xStart, yStart,*/ MANDEL_32_DARK, NB_ITERATIONS);
			}

	}

};


BigFloat2* BigMandel::xStart = new BigFloat2();
BigFloat2* BigMandel::yStart = new BigFloat2();
BigFloat2* BigMandel::x = new BigFloat2();
BigFloat2* BigMandel::y = new BigFloat2();
BigFloat2* BigMandel::xx = new BigFloat2();
BigFloat2* BigMandel::yy = new BigFloat2();
BigFloat2* BigMandel::temp = new BigFloat2();
BigFloat2* BigMandel::temp2 = new BigFloat2();