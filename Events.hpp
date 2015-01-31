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

using namespace std;


class Events{
public:

	// Zoom vers la cible du clic
	static void clicGauche(SDL_Event& event, Affichage* disp)
	{
		float x = disp->center.x + disp->scale*(((float)event.motion.x) / WIDTH - 0.5); // position du clic dans le plan compexe
		float y = disp->center.y + disp->scale*(((float)event.motion.y) / HEIGHT - 0.5);
		cout << "Old x = " << x << endl;
		disp->center.x = x + (disp->center.x - x) * ZOOM_FACTOR;
		disp->center.y = y + (disp->center.y - y) * ZOOM_FACTOR;
		disp->scale *= ZOOM_FACTOR;
		disp->ss = stringstream();
		disp->ss << "Centered in (" << disp->center.x << ", " << disp->center.y << ")";
		disp->start = chrono::system_clock::now();
			Mandelbrot::computeMandel(disp->pixels, WIDTH, HEIGHT, disp->center, disp->scale);
		disp->end = chrono::system_clock::now();
		disp->duration = disp->end - disp->start;
		cout << "Frame computing time : " << disp->duration.count() << endl;
		disp->dessin();
		SDL_SetWindowTitle(disp->win, disp->ss.str().c_str());
	}

	// Dezoome hors de la cible du clic
	static void clicDroit(SDL_Event& event, Affichage* disp)
	{
		float x = disp->center.x + disp->scale*(((float)event.motion.x) / WIDTH - 0.5); // position du clic dans le plan compexe
		float y = disp->center.y + disp->scale*(((float)event.motion.y) / HEIGHT - 0.5);
		cout << "Old x = " << x << endl;
		disp->center.x = x + (disp->center.x - x) / DEZOOM_FACTOR;
		disp->center.y = y + (disp->center.y - y) / DEZOOM_FACTOR;
		disp->scale /= DEZOOM_FACTOR;
		disp->ss = stringstream();
		disp->ss << "Centered in (" << disp->center.x << ", " << disp->center.y << ")";
		disp->start = chrono::system_clock::now();
			Mandelbrot::computeMandel(disp->pixels, WIDTH, HEIGHT, disp->center, disp->scale);
		disp->end = chrono::system_clock::now();
		disp->duration = disp->end - disp->start;
		cout << "Frame computing time : " << disp->duration.count() << endl;
		disp->dessin();
		SDL_SetWindowTitle(disp->win, disp->ss.str().c_str());
	}
};