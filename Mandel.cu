#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SDL2/SDL.h>

#include "Parametres.hpp"
#include "Affichage.hpp"



#define ASSERT(x, msg, retcode) \
    if (!(x)) \
			    { \
        cout << msg << " " << __FILE__ << ":" << __LINE__ << endl; \
        return retcode; \
			    }


__device__ static int iteratePoint(float xC, float yC, int nbIterations) {
	int count = 0;
	float x = 0;
	float y = 0;
	float xx = 0;
	float yy = 0;
	float temp;
	while (count < nbIterations && (xx + yy)  < 4.)
	{
		//z.mult(z);
		temp = x;
		x = xx - yy;
		y *= temp;
		y += y;
		x += xC;
		y += yC;
		count++;
		xx = x*x;
		yy = y*y;
	}
	if ((xx + yy) < 4.)
		return -1;
	return count;
}

// Convertit RGB en Uint32
__device__ static Uint32 couleur(int r, int g, int b)
{
	return (((r << 8) + g) << 8) + b;
}


__device__ static Uint32 computeColor_32_DARK(int countMax, int count, int divFactor)
{
	int k = ((count * divFactor) % countMax);
	int p = countMax / 4;
	int m = k % p;

	if (k < p)
		return couleur(0, 0, 255 * m / p);
	else if (k < 2 * p)
		return couleur(255 * m / p, 0, 255);
	else if (k < 3 * p)
		return couleur(255, 0, (255 - 255 * m / p));
	else return couleur(255, 255 * m / p, 0);

}


__global__ void computeMandel_GPU(uint32_t* result, float xCenter, float yCenter, float scale)
{
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

	if ((i >= WIDTH) || (j >= HEIGHT))
		return;

	int count = iteratePoint(xCenter + scale*(-0.5 + (float)i / WIDTH), (yCenter + scale*(-0.5 + (float)j / HEIGHT))*HEIGHT / ((float)WIDTH), NB_ITERATIONS);
	if (count == -1)
		result[j* WIDTH + i] = couleur(0, 0, 0);
	else
		result[j* WIDTH + i] = computeColor_32_DARK(NB_ITERATIONS, count, 1);
}



int affichageGPU(Affichage* disp)
{
	uint32_t *pixels_result;

	ASSERT(cudaSuccess == cudaMalloc(&pixels_result, WIDTH*HEIGHT * sizeof(uint32_t)), "Device allocation of pixel matrix failed", -1);

	dim3 cudaBlockSize(32, 32, 1);
	dim3 cudaGridSize((WIDTH - 1) / cudaBlockSize.x + 1, (HEIGHT - 1) / cudaBlockSize.y + 1);
	computeMandel_GPU << <cudaGridSize, cudaBlockSize >> >(pixels_result, disp->center.x, disp->center.y, disp->scale);

	ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
	ASSERT(cudaSuccess == cudaDeviceSynchronize(), "Kernel synchronization failed", -1);

	ASSERT(cudaSuccess == cudaMemcpy(disp->pixels, pixels_result, WIDTH*HEIGHT *sizeof(uint32_t), cudaMemcpyDeviceToHost), "Copy of pixel matrix from device to host failed", -1);

	ASSERT(cudaSuccess == cudaFree(pixels_result), "Device deallocation failed", -1);

	disp->dessin();

	return EXIT_SUCCESS;
}