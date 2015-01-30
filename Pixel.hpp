#pragma once

#include <SDL2/SDL.h>

class Pixel {



public:
	/* ********************************************************************* */
	/*obtenirPixel : permet de r�cup�rer la couleur d'un pixel
	Param�tres d'entr�e/sortie :
	SDL_Surface *surface : la surface sur laquelle on va r�cup�rer la couleur d'un pixel
	int x : la coordonn�e en x du pixel � r�cup�rer
	int y : la coordonn�e en y du pixel � r�cup�rer

	Uint32 resultat : la fonction renvoie le pixel aux coordonn�es (x,y) dans la surface
	*/
	Uint32 obtenirPixel(SDL_Surface *surface, int x, int y)
	{
		/*nbOctetsParPixel repr�sente le nombre d'octets utilis�s pour stocker un pixel.
		En multipliant ce nombre d'octets par 8 (un octet = 8 bits), on obtient la profondeur de couleur
		de l'image : 8, 16, 24 ou 32 bits.*/
		int nbOctetsParPixel = surface->format->BytesPerPixel;
		/* Ici p est l'adresse du pixel que l'on veut connaitre */
		/*surface->pixels contient l'adresse du premier pixel de l'image*/
		Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * nbOctetsParPixel;

		/*Gestion diff�rente suivant le nombre d'octets par pixel de l'image*/
		switch (nbOctetsParPixel)
		{
		case 1:
			return *p;

		case 2:
			return *(Uint16 *)p;

		case 3:
			/*Suivant l'architecture de la machine*/
			if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
				return p[0] << 16 | p[1] << 8 | p[2];
			else
				return p[0] | p[1] << 8 | p[2] << 16;

		case 4:
			return *(Uint32 *)p;

			/*Ne devrait pas arriver, mais �vite les erreurs*/
		default:
			return 0;
		}
	}


	/* ********************************************************************* */
	/*definirPixel : permet de modifier la couleur d'un pixel
	Param�tres d'entr�e/sortie :
	SDL_Surface *surface : la surface sur laquelle on va modifier la couleur d'un pixel
	int x : la coordonn�e en x du pixel � modifier
	int y : la coordonn�e en y du pixel � modifier
	Uint32 pixel : le pixel � ins�rer
	*/
	void definirPixel(SDL_Surface *surface, int x, int y, Uint32 pixel)
	{
		/*nbOctetsParPixel repr�sente le nombre d'octets utilis�s pour stocker un pixel.
		En multipliant ce nombre d'octets par 8 (un octet = 8 bits), on obtient la profondeur de couleur
		de l'image : 8, 16, 24 ou 32 bits.*/
		int nbOctetsParPixel = surface->format->BytesPerPixel;
		/*Ici p est l'adresse du pixel que l'on veut modifier*/
		/*surface->pixels contient l'adresse du premier pixel de l'image*/
		Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * nbOctetsParPixel;

		/*Gestion diff�rente suivant le nombre d'octets par pixel de l'image*/
		switch (nbOctetsParPixel)
		{
		case 1:
			*p = pixel;
			break;

		case 2:
			*(Uint16 *)p = pixel;
			break;

		case 3:
			/*Suivant l'architecture de la machine*/
			if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
			{
				p[0] = (pixel >> 16) & 0xff;
				p[1] = (pixel >> 8) & 0xff;
				p[2] = pixel & 0xff;
			}
			else
			{
				p[0] = pixel & 0xff;
				p[1] = (pixel >> 8) & 0xff;
				p[2] = (pixel >> 16) & 0xff;
			}
			break;

		case 4:
			*(Uint32 *)p = pixel;
			break;
		}
	}

};