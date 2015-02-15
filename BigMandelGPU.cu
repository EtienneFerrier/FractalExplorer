/*
Cette classe implémente le calcul de l'ensemble de Mandelbrot sur GPU.
*/

#pragma once

#include <cuda.h>
#include <device_functions.h>
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

// C = A + B
__device__ void add(bool posA, uint32_t* decA, bool posB, uint32_t* decB, bool* posC, uint32_t* decC)
{
	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;
	bool carry;

	decC[k] = decA[k] + decB[k];
	carry = decC[k] < decA[k];
	__syncthreads(); // Si souligné en rouge, n'est pas nécessairement une erreur

	if (k > 0)
	{
		decC[k - 1] += carry;
	}
}

__global__ void testKernel(bool posA, uint32_t* decA, bool posB, uint32_t* decB, bool* posC, uint32_t* decC)
{
	const unsigned int ti = threadIdx.x;
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

	//__shared__ uint32_t dec[32 * BIG_FLOAT_SIZE];

	if (i == 0)
	{
		add(posA, decA, posB, decB, posC, decC);
		*posC = true;
	}
	
}

// Fonction de communication avec le GPU, lance les thread et gère les échanges mémoire
int testBigMandelGPU()
{
	uint32_t* d_decA;
	uint32_t* d_decB;
	uint32_t* d_decC;
	bool* d_posA;
	bool* d_posB;
	bool* d_posC;

	ASSERT(cudaSuccess == cudaMalloc(&d_decA, BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decA failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decB, BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decB failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decC, BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decC failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posA, sizeof(bool)), "Device allocation of posA failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posB, sizeof(bool)), "Device allocation of posB failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posC, sizeof(bool)), "Device allocation of posC failed", -1);

	uint32_t h_decA[BIG_FLOAT_SIZE];
	uint32_t h_decB[BIG_FLOAT_SIZE];
	uint32_t h_decC[BIG_FLOAT_SIZE];
	bool h_posA, h_posB, h_posC;

	h_decA[0] = 1;
	h_decB[0] = 2;
	//h_decA[1] = 45;
	//h_decB[1] = 54;
	h_decA[1] = 4294967295;
	h_decB[1] = 4294967295;
	h_posA = true;
	h_posB = true;

	ASSERT(cudaSuccess == cudaMemcpy(d_decA, h_decA, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice), "Copy of decA from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_decB, h_decB, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice), "Copy of decB from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_posA, &h_posA, sizeof(bool), cudaMemcpyHostToDevice), "Copy of posA from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_posB, &h_posB, sizeof(bool), cudaMemcpyHostToDevice), "Copy of posB from host to device failed", -1);


	dim3 cudaBlockSize(32, 1, BIG_FLOAT_SIZE); // ATTENTION, 1024 threads max par block
	dim3 cudaGridSize(1, 1, 1);
	testKernel << <cudaGridSize, cudaBlockSize >> >(h_posA, d_decA, h_posB, d_decB, d_posC, d_decC);
	//nothing << <cudaGridSize, cudaBlockSize >> >();

	ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
	ASSERT(cudaSuccess == cudaDeviceSynchronize(), "Kernel synchronization failed", -1);




	ASSERT(cudaSuccess == cudaMemcpy(h_decC, d_decC, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost), "Copy of decC from device to host failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(&h_posC, d_posC, sizeof(bool), cudaMemcpyDeviceToHost), "Copy of posC from device to host failed", -1);



	ASSERT(cudaSuccess == cudaFree(d_decA), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decB), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decC), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posA), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posB), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posC), "Device deallocation failed", -1);


	cout << "C = " << h_posC << "    " << h_decC[0] << " " << h_decC[1] << endl;

	return EXIT_SUCCESS;
}