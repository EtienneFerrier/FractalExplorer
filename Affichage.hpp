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
#include "Complexe.hpp"

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
	Complexe center = Complexe(0., 0.);		// Centre de l'image calculee. Initialement egal a (0, 0))
	float scale = 4.;						// Largeur de l'image calculee. Initialement egale a 4

	Affichage(){
		pixels = new Uint32[WIDTH*HEIGHT];
	}

	~Affichage()
	{
		delete[] pixels;
	}

	// Ferme et libere l'affichage SDL
	void fermer()
	{
		SDL_DestroyTexture(tex);
		SDL_DestroyRenderer(ren);
		SDL_DestroyWindow(win);
		SDL_Quit();
	}

	// Ouvre et initialise la fenetre SDL
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

	// Met a jour le titre de la fenetre avec des informations sur la zone exploree du plan complexe
	void majTitre()
	{
		ss = stringstream();
		ss << "Centered in (" << center.x << ", " << center.y << "). Width : " << scale;
		SDL_SetWindowTitle(win, ss.str().c_str());
	}

	// Affiche dans la fenetre
	void dessin()
	{
		majTitre();

		tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
			SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

		SDL_UpdateTexture(tex, NULL, pixels, WIDTH * sizeof(Uint32));
		SDL_RenderCopy(ren, tex, NULL, NULL);
		SDL_RenderPresent(ren);
		SDL_DestroyTexture(tex);
	}
	
};

