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
		switch (methode)
		{
		case BIN_NB:
			Complexe z0(x, y);
			Complexe z(x, y);
			int count = 0;
			while (count < nbIterations && z.squaredNorm() < 4.)
			{
				z.mult(z);
				z.add(z0);
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
	static void computeMandel(Uint32* result, int width, int height)
	{
		for (int i = 0; i < width; i++)
			for (int j = 0; j < height; j++)
			{
			result[i*height + j] = computeColor(-2. + 4.*((float)i / width), -2. + 4.*((float)j / height), BIN_NB, 10);
			}

	}

};