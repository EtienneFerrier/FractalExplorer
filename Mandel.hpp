/*
Cette classe implémente le calcul de l'ensemble de Mandelbrot sur CPU.
L'affichage n'est pas géré.
Optimisation possible.
*/

#pragma once

#include <SDL2/SDL.h>
#include "Complexe.hpp"
#include <iostream>


#define BIN_NB 1
#define CERCLE_NB 2

class Mandelbrot {

public:

	// Convertit RGB en Uint32
	static Uint32 couleur(int r, int g, int b)
	{
		return (((r << 8) + g) << 8) + b;
	}

	// Calcule la couleur d'un point de l'ensemble de Mandelbrot en fonction d'une méthode de coloration et d'un nombre d'itérations.
	static Uint32 computeColor(float x, float y, int methode, int nbIterations)
	{
		int count;
		Complexe c(x, y);
		Complexe z(0., 0.);
		switch (methode)
		{
		case CERCLE_NB:
			if (z.squaredNorm() > 2.)
				return couleur(0, 0, 0);
			else return couleur(255, 255, 255);
			break;
		case BIN_NB:
			count = 0;
			while (count < nbIterations && z.squaredNorm() < 4.)
			{
				z.mult(z);
				z.add(c);
				count++;
			}
			if (z.squaredNorm() < 4.)
				return couleur(0, 0, 0);
			else return couleur(255, 255, 255);
			break;
		default:
			std::cout << "Pas de methode de coloration" << std::endl;
			break;
		}
	}

	// Methode de test.
	// Calcule l'ensemble de Mandelbrot sur le carre [-2, 2]x[-2, 2] avec une coloration N&B (10 itérations).
	static void computeMandel(Uint32* result, int width, int height, Complexe& center, float scale)
	{
		for (int i = 0; i < width; i++)
			for (int j = 0; j < height; j++)
			{
				//result[j*width + i] = computeColor(-2. + 4.*((float)i / width), (-2. + 4.*((float)j / height))*height / ((float)width), BIN_NB, 50);
				result[j*width + i] = computeColor(center.x + scale*(-0.5+(float)i / width), (center.y + scale*(-0.5+(float)j / height))*height / ((float)width), BIN_NB, 50);
			}

	}

};