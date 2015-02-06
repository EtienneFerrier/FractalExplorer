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

BigFloat* bigScale;

class Events{
public:

	//
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
				BigFloat xCenter(0);
				BigFloat yCenter(0);
				BigMandel::computeMandel(disp->pixels, xCenter, yCenter, *bigScale);
			}

			disp->end = chrono::system_clock::now();
			disp->duration = disp->end - disp->start;
			cout << "Frame computing time : " << disp->duration.count() << endl;

			// Affichage de la fractale
			disp->dessin();
	}

	//static BigFloat* bigZoomFactor;
	// Zoom vers la cible du clic
	static void clicGauche(SDL_Event& event, Affichage* disp)
	{
		// Calcul de la position du clic dans le plan complexe
		float x = disp->center.x + disp->scale*(((float)event.motion.x) / WIDTH - 0.5f); 
		float y = disp->center.y + disp->scale*(((float)event.motion.y) / HEIGHT - 0.5f); 
		//cout << "Old x = " << x << endl;

		// MAJ de la position du nouveau centre dans le plan complexe
		disp->center.x = x + (disp->center.x - x) * ZOOM_FACTOR; 
		disp->center.y = y + (disp->center.y - y) * ZOOM_FACTOR;
		
		// MAJ de l'echelle
		disp->scale *= ZOOM_FACTOR;
		bigScale = new BigFloat(disp->scale);
		
		//Nouveau calcul de la fractale avec chrono
		disp->start = chrono::system_clock::now();

		if (GPU)
			affichageGPU(disp);
		else
			if (BIG_FLOAT_SIZE == 0)
				Mandelbrot::computeMandel(disp->pixels, disp->center, disp->scale);
			else {
				BigFloat xCenter(disp->center.x);
				BigFloat yCenter(disp->center.y);
				BigMandel::computeMandel(disp->pixels, xCenter, yCenter, *bigScale);
			}

		disp->end = chrono::system_clock::now();
		disp->duration = disp->end - disp->start;
		cout << "Frame computing time : " << disp->duration.count() << endl;
		
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