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
#include "BigFloat2.hpp"



//__global__ void computeMandel_GPU(uint32_t* result, float xCenter, float yCenter, float scale);

using namespace std;





int affichageGPU(Affichage* disp);

// TESTS:
// (-1+0.9)^2 = 0.01
// (-1+0.99)^2 = 0.0001

int main(int argc, char** argv)
{
	BigFloat2 a, b, c, d;
	a.pos = 1;
	b.pos = 0;
	a[0] = 0;
	b[0] = 0;
	a[1] = 2;
	b[1] = 0xffffffff;
	a[2] = 0;
	b[2] = 0;
	a[3] = 0;
	b[3] = 0;
	BigFloat2::add(a, b, c);
	BigFloat2::mult(a, b, d);
	cout << "a = " << a.pos << "    " << a[0] << " " << a[1] << " " << a[2] << " " << a[3] << endl;
	cout << "b = " << b.pos << "    " << b[0] << " " << b[1] << " " << b[2] << " " << b[3] << endl;
	cout << "c = " << c.pos << "    " << c[0] << " " << c[1] << " " << c[2] << " " << c[3] << endl;
	cout << "d = " << d.pos << "    " << d[0] << " " << d[1] << " " << d[2] << " " << d[3] << endl;
	while (1);
	return 0;
}

//int main(int argc, char** argv)
//{
//
//
//	BigFloat y(0, 2166, 3938, 8437, 7127, 0, 0);
//	BigFloat::negate(y);
//	BigFloat x(0, 3750, 0012, 0061, 8655, 0, 0);
//	x.display();
//	y.display();
//
//
//	Affichage display;
//	if(display.initSDLAffichage() < 0)
//		return 0;
//
//	/* Calcul de la fractale */ 
//	Events::initialDisplay(&display);
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

