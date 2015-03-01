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
#include "BigFloat2.hpp"
#include "BigMandel.hpp"

using namespace std;


int affichageGPU(Affichage* disp);
int computeBigMandelGPU(Affichage* disp);
int computeBigMandelGPU(Affichage* disp, bool h_posCx, uint32_t* h_decCx, bool h_posCy, uint32_t* h_decCy, uint32_t* h_decS);

class Events{
public:
	static BigFloat2* bigScale;
	static BigFloat2* xCenter;
	static BigFloat2* yCenter;
	
	static void initialDisplay(Affichage* disp) {
		bigScale = new BigFloat2(disp->scale);

		//Nouveau calcul de la fractale avec chrono
		disp->start = chrono::system_clock::now();

		if (GPU && BIG_FLOAT_SIZE == 0)
		{
			affichageGPU(disp);		
		}
		else if (GPU)
		{
			Events::yCenter = new BigFloat2();
			Events::xCenter = new BigFloat2();
			//computeBigMandelGPU(disp);
			computeBigMandelGPU(disp, xCenter->pos, xCenter->decimals, yCenter->pos, yCenter->decimals, bigScale->decimals);
		}
		else
			if (BIG_FLOAT_SIZE == 0)
				Mandelbrot::computeMandel(disp->pixels, disp->center, disp->scale);
			else {
				if (INTERACTIVE) {
					BigFloat2 xCenter(0);
					BigFloat2 yCenter(0);
				}
				else {
					Events::yCenter = new BigFloat2(true, 0, 121012162, 3888660452, 0);
					Events::xCenter = new BigFloat2(false, 1, 3178543730, 764955228, 0);
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
		BigFloat2 temp, temp2; 
		BigFloat2::mult(((float)event.motion.x) / WIDTH - 0.5f, *bigScale, temp2);
		BigFloat2::add(*xCenter, temp2, temp);
		temp2.reset();
		BigFloat2::mult((1.f - ZOOM_FACTOR), temp, temp2);
		temp.reset();
		BigFloat2::mult(ZOOM_FACTOR, *xCenter, temp);
		BigFloat2::add(temp2, temp, *xCenter);

		temp.reset();
		temp2.reset();
		BigFloat2::mult(((float)event.motion.y) / HEIGHT - 0.5f, *bigScale, temp2);
		BigFloat2::add(*yCenter, temp2, temp);
		temp2.reset();
		BigFloat2::mult((1.f - ZOOM_FACTOR), temp, temp2);
		temp.reset();
		BigFloat2::mult(ZOOM_FACTOR, *yCenter, temp);
		BigFloat2::add(temp2, temp, *yCenter);
	}

	//static BigFloat2* bigZoomFactor;
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
		//BigFloat2 zoomFactor(true, 0, 0x80000000, 0, 0);
		BigFloat2 temp, temp2;
		BigFloat2::mult(ZOOM_FACTOR, *bigScale, temp);
		bigScale->reset();
		temp2.copy(*bigScale);
		BigFloat2::add(temp, temp2, *bigScale);
		
		//Nouveau calcul de la fractale avec chrono
		disp->start = chrono::system_clock::now();

		if (GPU && BIG_FLOAT_SIZE == 0)
			affichageGPU(disp);
		else if (GPU)
			computeBigMandelGPU(disp, xCenter->pos, xCenter->decimals, yCenter->pos, yCenter->decimals, bigScale->decimals);
		else
			if (BIG_FLOAT_SIZE == 0)
				Mandelbrot::computeMandel(disp->pixels, disp->center, disp->scale);
			else {
				BigMandel::computeMandel(disp->pixels, *xCenter, *yCenter, *bigScale);
			}

		disp->end = chrono::system_clock::now();
		disp->duration = disp->end - disp->start;
		cout << "Frame computing time : " << disp->duration.count() << endl;
		//cout << "Frame computing scale : " << disp->scale << endl;
		
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
		if (GPU && BIG_FLOAT_SIZE == 0)
			affichageGPU(disp);
		else if (GPU)
			computeBigMandelGPU(disp, xCenter->pos, xCenter->decimals, yCenter->pos, yCenter->decimals, bigScale->decimals);
		else
			Mandelbrot::computeMandel(disp->pixels, disp->center, disp->scale);
		disp->end = chrono::system_clock::now();
		disp->duration = disp->end - disp->start;
		cout << "Frame computing time : " << disp->duration.count() << endl;
		disp->dessin();
	}
};

BigFloat2* Events::bigScale = new BigFloat2();
BigFloat2* Events::xCenter = new BigFloat2();
BigFloat2* Events::yCenter = new BigFloat2();