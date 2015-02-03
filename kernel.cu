#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <SDL2/SDL.h>

#include "Parametres.hpp"
#include "Pixel.hpp"
#include "Mandel.hpp"
#include "Events.hpp"
#include "Affichage.hpp"
#include "BigFloat.hpp"



//__global__ void computeMandel_GPU(uint32_t* result, float xCenter, float yCenter, float scale);

using namespace std;





int affichageGPU(Affichage* disp);


int main(int argc, char** argv)
{
	BigFloat big;
	big.decimals[1] = 1;
	big.display();
	while (1);
	return 0;
}

//int main(int argc, char** argv)
//{
//	Affichage display;
//	if(display.initSDLAffichage() < 0)
//		return 0;
//
//	/* Calcul de la fractale */ 
//	//Mandelbrot::computeMandel(display.pixels, WIDTH, HEIGHT, display.center, display.scale);
//
//	affichageGPU(&display);
//	
//
//	/* Affichage de la fractale */
//	//display.dessin();
//
//	/* Boucle des evenements */
//	bool quit = false;
//	SDL_Event event;
//
//	while (!quit)
//	{
//		SDL_WaitEvent(&event);
//		bool buttonDown;
//		switch (event.type)
//		{
//		case SDL_MOUSEBUTTONDOWN:
//			switch (event.button.button)
//			{
//			case SDL_BUTTON_LEFT:
//				buttonDown = true;
//				Events::clicGauche(event, &display);
//				while (buttonDown)
//				{
//					SDL_PumpEvents();
//					
//					if (SDL_GetMouseState(&(event.button.x), &(event.button.y)) & SDL_BUTTON(SDL_BUTTON_LEFT)) {
//						Events::clicGauche(event, &display);
//					} else {
//						buttonDown = false;
//					}
//
//				}
//				break;
//			case SDL_BUTTON_RIGHT:
//				buttonDown = true;
//				Events::clicDroit(event, &display);
//				while (buttonDown)
//				{
//					SDL_PumpEvents();
//
//					if (SDL_GetMouseState(&(event.button.x), &(event.button.y)) & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
//						Events::clicDroit(event, &display);
//					}
//					else {
//						buttonDown = false;
//					}
//
//				}
//				break;
//			default:
//				SDL_ShowSimpleMessageBox(0, "Mouse", "Some other button was pressed!", display.win);
//				break;
//			}
//			break;
//		case SDL_QUIT:
//			quit = true;
//			break;
//		}
//
//	}
//
//	return 0;
//}

