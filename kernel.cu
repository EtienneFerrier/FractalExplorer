#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <SDL2/SDL.h>

#include "Pixel.hpp"
#include "Mandel.hpp"

#define WIDTH 800
#define HEIGHT 600

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


Uint32 couleur(int r, int g, int b)
{
	return (((r << 8) + g) << 8) + b;
}

void dessin(SDL_Renderer * ren, SDL_Texture * tex, Uint32 * pixels, float alpha)
{
	//Uint32  *p;
	//Uint8 r, g, b;
	//int x, y;
	//float beta = 1 - alpha;
	//pixels = (Uint32*)malloc(WIDTH*HEIGHT*sizeof(Uint32));
	//if (!pixels) { fprintf(stderr, "Erreur allocation\n"); return; }

	tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
		SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

	SDL_UpdateTexture(tex, NULL, pixels, WIDTH * sizeof(Uint32));
	SDL_RenderCopy(ren, tex, NULL, NULL);
	SDL_RenderPresent(ren);
	SDL_DestroyTexture(tex);
	//free(pixels);
}


int main(int argc, char** argv)
{
	SDL_Window *win = 0;
	SDL_Renderer *ren = 0;
	Uint32 * pixels = 0;
	SDL_Texture * tex = 0;
	float alpha = 0.0;
	float pas = 0.03;
	int n;
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

	pixels = (Uint32*)malloc(WIDTH*HEIGHT*sizeof(Uint32));
	if (!pixels) { fprintf(stderr, "Erreur allocation\n"); return -1; }

	tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888,
		SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

	Complexe center(0,0);
	float scale = 4;
	float zoomFactor = 0.5;
	Mandelbrot::computeMandel(pixels, WIDTH, HEIGHT, center, scale);

	dessin(ren, tex, pixels, alpha);

	// handle events

	bool quit = false;
	SDL_Event event;
	stringstream ss;
	int mouseX;
	int mouseY;
	float x, y;
	chrono::time_point<std::chrono::system_clock> start, end;
	chrono::duration<double> duration;
	while (!quit)
	{
		SDL_WaitEvent(&event);

		switch (event.type)
		{
		case SDL_MOUSEBUTTONDOWN:
			switch (event.button.button)
			{
			case SDL_BUTTON_LEFT:
				mouseX = event.motion.x;
				mouseY = event.motion.y;
				x = center.x + scale*(((float)mouseX) / WIDTH - 0.5);
				y = center.y + scale*(((float)mouseY) / HEIGHT - 0.5);
				cout << "Old x = " << x << endl;
				center.x = x + (center.x - x) * zoomFactor;
				center.y = y + (center.y - y) * zoomFactor;
				scale *= zoomFactor;
				ss = stringstream();
				ss << "Centered in " << center.x << " + i" << center.y;
				start = chrono::system_clock::now();
				Mandelbrot::computeMandel(pixels, WIDTH, HEIGHT, center, scale);
				end = chrono::system_clock::now();
				duration = end - start;
				cout << "Frame computing time : " << duration.count() << endl;
				dessin(ren, tex, pixels, alpha);
				SDL_SetWindowTitle(win, ss.str().c_str());
				break;
			case SDL_BUTTON_RIGHT:
				SDL_ShowSimpleMessageBox(0, "Mouse", "Right button was pressed!", win);
				break;
			default:
				SDL_ShowSimpleMessageBox(0, "Mouse", "Some other button was pressed!", win);
				break;
			}
			break;
		case SDL_QUIT:
			quit = true;
			break;
		}

	}

	SDL_DestroyTexture(tex);
	SDL_DestroyRenderer(ren);
	SDL_DestroyWindow(win);
	SDL_Quit();
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<< 1, size>>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}