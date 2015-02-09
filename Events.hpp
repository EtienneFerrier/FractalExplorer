/*
Cette classe permet de gérer les evenements a la souris et au clavier.
Elle fait l'interface entre la boucle d'evenements et les methodes de calcul et d'affichage des fractales.
*/

#pragma once

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <SDL2/SDL.h>

#include "Parametres.hpp"
#include "Mandel.hpp"
#include "Affichage.hpp"
#include "BigFloat.hpp"
#include "BigMandel.hpp"

using namespace std;


int affichageGPU(Affichage* disp);



class Events{
public:
	static BigFloat* bigScale;
	static BigFloat* xCenter;
	static BigFloat* yCenter;
	
	static void initialDisplay(Affichage* disp) {
		bigScale = new BigFloat(disp->scale);

		//Nouveau calcul de la fractale avec chrono
		disp->start = chrono::system_clock::now();

		if (GPU)
			affichageGPU(disp);
		else
			if (BIG_FLOAT_SIZE == 0)
				Mandelbrot::computeMandel(disp->pixels, disp->center, disp->scale);
			else {
				if (INTERACTIVE) {
					BigFloat xCenter(0);
					BigFloat yCenter(0);
				}
				else {
					//BigFloat temp(0, 2166, 3938, 8437, 7127, 0, 0);
					//BigFloat temp(1, 7400, 6238, 2579, 3399, 0, 0);
					BigFloat temp(1, 3178543730, 764955228, 0, 0);
					BigFloat::negate(temp);
					//Events::xCenter = new BigFloat(0, 3750, 0012, 0061, 8655, 0, 0);
					//Events::yCenter = new BigFloat(0, 281, 7533, 9779, 2110, 0, 0);
					Events::yCenter = new BigFloat(0, 121012162, 3888660452, 0, 0);
					Events::xCenter = new BigFloat(temp);
				}
				BigMandel::computeMandel(disp->pixels, *xCenter, *yCenter, *bigScale);
			}

			disp->end = chrono::system_clock::now();
			disp->duration = disp->end - disp->start;
			cout << "Frame computing time : " << disp->duration.count() << endl;

			// Affichage de la fractale
			disp->dessin();
	}

	static void updateBigCenter(SDL_Event& event) {
		BigFloat temp, temp2; 
		BigFloat::mult(((float)event.motion.x) / WIDTH - 0.5f, *bigScale, temp);
		BigFloat::add(*xCenter, temp);
		BigFloat::mult((1 - ZOOM_FACTOR), temp, temp2);
		temp.reset();
		BigFloat::mult(ZOOM_FACTOR, *xCenter, temp);
		BigFloat::add(temp2, temp, *xCenter);

		temp.reset();
		temp2.reset();
		BigFloat::mult(((float)event.motion.y) / HEIGHT - 0.5f, *bigScale, temp);
		BigFloat::add(*yCenter, temp);
		BigFloat::mult((1 - ZOOM_FACTOR), temp, temp2);
		temp.reset();
		BigFloat::mult(ZOOM_FACTOR, *yCenter, temp);
		BigFloat::add(temp2, temp, *yCenter);
	}

	//static BigFloat* bigZoomFactor;
	// Zoom vers la cible du clic
	static void clicGauche(SDL_Event& event, Affichage* disp)
	{

		// Calcul de la position du clic dans le plan complexe
		float x = disp->center.x + disp->scale*(((float)event.motion.x) / WIDTH - 0.5f);
		float y = disp->center.y + disp->scale*(((float)event.motion.y) / HEIGHT - 0.5f);

		// MAJ de la position du nouveau centre dans le plan complexe
		if (INTERACTIVE) {
			disp->center.x = x + (disp->center.x - x) * ZOOM_FACTOR;
			disp->center.y = y + (disp->center.y - y) * ZOOM_FACTOR;
			updateBigCenter(event);
		}
		

		// MAJ de l'echelle
		disp->scale *= ZOOM_FACTOR;
		BigFloat zoomFactor(ZOOM_FACTOR);
		BigFloat temp;
		BigFloat::mult(zoomFactor, *bigScale, temp);
		bigScale->reset();
		BigFloat::add(temp, *bigScale);
		
		//Nouveau calcul de la fractale avec chrono
		disp->start = chrono::system_clock::now();

		if (GPU)
			affichageGPU(disp);
		else
			if (BIG_FLOAT_SIZE == 0)
				Mandelbrot::computeMandel(disp->pixels, disp->center, disp->scale);
			else {
				BigMandel::computeMandel(disp->pixels, *xCenter, *yCenter, *bigScale);
			}

		disp->end = chrono::system_clock::now();
		disp->duration = disp->end - disp->start;
		cout << "Frame computing time : " << disp->duration.count() << endl;
		cout << "Frame computing scale : " << disp->scale << endl;
		
		// Affichage de la fractale
		disp->dessin();
	}

	// Dezoome hors de la cible du clic
	static void clicDroit(SDL_Event& event, Affichage* disp)
	{
		/* Pour les commentaires, voir la methode Event::clicGauche */
		float x = disp->center.x + disp->scale*(((float)event.motion.x) / WIDTH - 0.5f); 
		float y = disp->center.y + disp->scale*(((float)event.motion.y) / HEIGHT - 0.5f); 
		//cout << "Old x = " << x << endl;
		disp->center.x = x + (disp->center.x - x) / DEZOOM_FACTOR;
		disp->center.y = y + (disp->center.y - y) / DEZOOM_FACTOR;
		disp->scale /= DEZOOM_FACTOR;
		disp->start = chrono::system_clock::now();
		if (GPU)
			affichageGPU(disp);
		else
			Mandelbrot::computeMandel(disp->pixels, disp->center, disp->scale);
		disp->end = chrono::system_clock::now();
		disp->duration = disp->end - disp->start;
		cout << "Frame computing time : " << disp->duration.count() << endl;
		disp->dessin();
	}
};

BigFloat* Events::bigScale = new BigFloat();
BigFloat* Events::xCenter = new BigFloat();
BigFloat* Events::yCenter = new BigFloat();