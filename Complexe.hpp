/*
Cette classe implémente le calcul complexe sur CPU.
Optimisations possibles.
*/


#pragma once

class Complexe {
public:
	float x;
	float y;

	Complexe(float x0, float y0)
	{
		x = x0;
		y = y0;
	}

	Complexe()
	{
		x = 0.;
		y = 0.;
	}

	// Addition en place
	void add(Complexe c)
	{
		x += c.x;
		y += c.y;
	}

	// Soustraction en place
	void sub(Complexe c)
	{
		x -= c.x;
		y -= c.y;
	}

	// Multiplication en place
	void mult(Complexe c)
	{
		float tmpx = x;
		x = x*c.x - y*c.y;
		y = y*c.x + tmpx*c.y;
	}

	// Norme euclidienne au carre
	float squaredNorm()
	{
		return (x*x + y*y);
	}

};