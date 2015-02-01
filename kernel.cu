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



__global__ void computeMandel_GPU(uint32_t* result, float xCenter, float yCenter, float scale);

using namespace std;




cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}



//int main(int argc, char* argv[])
//{
//	//const int arraySize = 5;
//	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
//	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
//	//int c[arraySize] = { 0 };
//
//	//// Add vectors in parallel.
//	//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//	//if (cudaStatus != cudaSuccess) {
//	//	fprintf(stderr, "addWithCuda failed!");
//	//	return 1;
//	//}
//
//	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//	//	c[0], c[1], c[2], c[3], c[4]);
//
//	//// cudaDeviceReset must be called before exiting in order for profiling and
//	//// tracing tools such as Nsight and Visual Profiler to show complete traces.
//	//cudaStatus = cudaDeviceReset();
//	//if (cudaStatus != cudaSuccess) {
//	//	fprintf(stderr, "cudaDeviceReset failed!");
//	//	return 1;
//	//}
//
//	//============================================================ Partie DSL =========================================
//	// Notre fenêtre
//
//	SDL_Window* fenetre(0);
//	SDL_Event evenements;
//	bool terminer(false);
//
//
//	// Initialisation de la SDL
//
//	if (SDL_Init(SDL_INIT_VIDEO) < 0)
//	{
//		std::cout << "Erreur lors de l'initialisation de la SDL : " << SDL_GetError() << std::endl;
//		SDL_Quit();
//
//		return -1;
//	}
//
//
//	// Création de la fenêtre
//
//	cout << "Go Fenetre" << endl;
//
//	fenetre = SDL_CreateWindow("Test SDL 2.0", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_SHOWN);
//
//	SDL_Surface* surface = SDL_GetWindowSurface(fenetre);
//
//	const SDL_PixelFormat* format = surface->format;
//	
//	Uint32 pixel = SDL_MapRGB(format, 100, 100, 100);
//
//	//Uint8 r, g, b, a;
//
//
//	//SDL_LockSurface(surface); /*On bloque la surface*/
//
//

int affichageGPU(Affichage* disp);

int main(int argc, char** argv)
{
	Affichage display;
	if(display.initSDLAffichage() < 0)
		return 0;

	/* Calcul de la fractale */ 
	//Mandelbrot::computeMandel(display.pixels, WIDTH, HEIGHT, display.center, display.scale);

	affichageGPU(&display);
	

	/* Affichage de la fractale */
	//display.dessin();

	/* Boucle des evenements */
	bool quit = false;
	SDL_Event event;

	while (!quit)
	{
		SDL_WaitEvent(&event);
		bool buttonDown;
		switch (event.type)
		{
		case SDL_MOUSEBUTTONDOWN:
			switch (event.button.button)
			{
			case SDL_BUTTON_LEFT:
				buttonDown = true;
				Events::clicGauche(event, &display);
				while (buttonDown)
				{
					SDL_PumpEvents();
					
					if (SDL_GetMouseState(&(event.button.x), &(event.button.y)) & SDL_BUTTON(SDL_BUTTON_LEFT)) {
						Events::clicGauche(event, &display);
					} else {
						buttonDown = false;
					}

				}
				break;
			case SDL_BUTTON_RIGHT:
				buttonDown = true;
				Events::clicDroit(event, &display);
				while (buttonDown)
				{
					SDL_PumpEvents();

					if (SDL_GetMouseState(&(event.button.x), &(event.button.y)) & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
						Events::clicDroit(event, &display);
					}
					else {
						buttonDown = false;
					}

				}
				break;
			default:
				SDL_ShowSimpleMessageBox(0, "Mouse", "Some other button was pressed!", display.win);
				break;
			}
			break;
		case SDL_QUIT:
			quit = true;
			break;
		}

	}

	return 0;
}

