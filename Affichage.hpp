/*
Cette classe contient toutes les informations qui permettent l'affichage indépendamment de la fractale calculée.

Elle contient donc :
- Les objets SDL d'affichage
- Les objets de mesure de temps
- Les objets qui expriment la zone du plan complexe explorée

Ainsi que :
- Une methode d'initialisation de l'affichage
- Une methode d'affichage d'une image calculee
*/


#pragma once

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <SDL2/SDL.h>

#include "Parametres.hpp"

using namespace std;

class Affichage{
public:

	/* Objets d'affichage SDL */
	SDL_Window *win = 0;
	SDL_Renderer *ren = 0;	
	SDL_Texture * tex = 0;
	float alpha = 0.0;

	/* Outils de mesure du temps */
	stringstream ss;
	chrono::time_point<std::chrono::system_clock> start;
	chrono::time_point<std::chrono::system_clock> end;
	chrono::duration<double> duration;

	/* Pour le stockage des images calculees */
	Uint32* pixels = 0;

	/* Parametres de la zone du plan complexe exploree */
	Complexe center;		// Centre de l'image calculee
	float scale;			// Largeur de l'image calculee


	Affichage()
	{
		center = Complexe(0., 0.);
		scale = 4.;
	}

	~Affichage()
	{
		free(pixels);
	}

	int initSDLAffichage()
	{
		/* Initialisation de la SDL. Si ça se passe mal, on quitte */
		if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
		{
			fprintf(stderr, "Erreur initialisation\n");
			return -1;
		}

		/* Création de la fenêtre et du renderer */
		SDL_CreateWindowAndRenderer(WIDTH, HEIGHT, 0, &win, &ren);

		if (!win || !ren)
		{
			fprintf(stderr, "Erreur à la création des fenêtres\n");
			SDL_Quit();
			return -1;
		}

		/* Affichage du fond noir */
		SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
		SDL_RenderClear(ren);
		SDL_RenderPresent(ren);

		/* Création de la surface qui sera affichee a l'ecran */
		pixels = (Uint32*)malloc(WIDTH*HEIGHT*sizeof(Uint32));
		if (!pixels) { fprintf(stderr, "Erreur allocation\n"); return -1; }

		tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
			SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

		return 0;
	}

	void dessin()
	{
		tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
			SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

		SDL_UpdateTexture(tex, NULL, pixels, WIDTH * sizeof(Uint32));
		SDL_RenderCopy(ren, tex, NULL, NULL);
		SDL_RenderPresent(ren);
		SDL_DestroyTexture(tex);
	}
	
};

