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
__device__ void add(/*reg*/bool posA, /*shared*/uint32_t* decA, /*reg*/bool posB, /*shared*/uint32_t* decB, /*shared*/bool* posC, /*shared*/uint32_t* decC)
{
	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;
	bool carry;

	if (posA == posB)	// Cas simple (addition unsigned)
	{
		decC[k] = decA[k] + decB[k]; // Addition
		carry = decC[k] < decA[k]; // Calcul des retenues
		__syncthreads();

		if (k > 0) 
		{
			decC[k - 1] += carry; // Propagation des retenues (une seule fois)
		}
		else
		{
			*posC = posA; // Calcul du signe du resultat
		}
		__syncthreads();
	}
	else // Cas complique (soustraction)
	{
		decC[k] = decA[k] - decB[k]; // Soustraction
		carry = decC[k] > decA[k]; // Calcul des retenues 
		__syncthreads();

		if (k > 0)
		{
			decC[k-1] -= carry; // Propagation des retenues (une seule fois)
		}
		__syncthreads();

		if(k == 0)
		{
			*posC = (carry || decC[0] == -1) ^ posA; // ATTENTION : pas le signe de C mais un indicateur de |A| < |B| (valeur absolue)
		}
		__syncthreads();

		if (*posC ^ posA) // Dans ce cas, |B| > |A|. On doit donc recalculer la soustraction
		{
			decC[k] = decB[k] - decA[k];
			carry = decC[k] > decB[k];
			__syncthreads();

			if (k > 0)
			{
				decC[k - 1] -= carry;
			}
			else
			{
				*posC = posB; // Signe du resultat
			}
			__syncthreads();
		}
	}
}

// A = A + B (In Place)
__device__ void addIP(/*shared*/bool* posA, /*shared*/uint32_t* decA, /*shared*/bool *posB, /*shared*/uint32_t* decB)
{
	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

	uint32_t tmp; // Sera ajoute a posA[k]
	bool carry; // Sera ajoute a posA[k-1]

	if (*posA == *posB)	// Cas simple (addition unsigned)
	{
		tmp = decA[k] + decB[k];
		carry = tmp < decA[k];

		decA[k] = tmp;
		__syncthreads();

		if (k > 0)
		{
			decA[k - 1] += carry; // NON EXACT : Propagation des retenues une seule fois
		}

		__syncthreads();
	}
	else // Cas complique (soustraction)
	{
		tmp = decA[k] - decB[k]; // Soustraction
		carry = tmp > decA[k]; // Calcul des retenues 

		if (k == 1)
		{
			*posA = ((decA[0] < decB[0]) || ((decA[0] == decB[0]) && carry)) ^ *posA; // Bottle-neck et NON EXACT (signe donné par la soustraction des 2 premiers bits)
		}
		__syncthreads();

		if(*posA == *posB) // Dans ce cas, |B| > |A|. On doit donc recalculer la soustraction
		{
			tmp = -tmp; // = decB[k] - decA[k]
			carry = !(tmp == 0 || carry); // = tmp > decB[k]
		}

		decA[k] = tmp;
		__syncthreads();
		
		if (k > 0)
			decA[k - 1] -= carry;
		__syncthreads();
	}
}

//Fonction servant à multiplier deux chiffres
/*	a (reg) et b (reg) sont les copies des decimaux a (shared) et b (shared)
	tmpLittle (reg) est un tas qui contient la somme partielle a ajouter plus tard a little (shared) 
	tmpBig (reg) est un tas qui contient la somme partielle a ajouter plus tard a big (shared) 
	carry (reg) est un tas qui contient la somme des retenues a ajouter plus tard au decimal avant big (shared)
	Chaque appel demande une lecture SHARED -> REG de 2 cases au préalable. Possibilité d'optimiser.*/
__device__ void multDigDig(/*reg*/uint32_t a, /*reg*/uint32_t b, /*reg*/uint8_t* carry, /*reg*/uint32_t* tmpBig, /*reg*/uint32_t* tmpLittle) {

	uint32_t mask = 0xFFFF;
	uint32_t midh, midl, ahbh, ahbl, albh, albl, temp;

	// Ces produits ne peuvent pas faire d'overflow
	// TODO: certaines mises en mémoire sont inutiles
	ahbh = (a >> 16) * (b >> 16);
	ahbl = (a >> 16) * (mask & b);
	albh = (mask & a) * (b >> 16);
	albl = (mask & a) * (mask & b);

	// Le carry est ce qu'on ajoute au chiffre supérieur à la fin
	midl = ahbl + albh;
	temp = ((midl < ahbl) << 16);
	*tmpBig += temp;
	*carry += (*tmpBig < temp);

	// On coupe le middle
	midh = midl >> 16;
	midl = midl << 16;

	*tmpLittle += albl;
	*tmpBig += (*tmpLittle < albl); //NON EXACT (pas de propagation)

	*tmpLittle += midl;
	*tmpBig += (*tmpLittle < midl); //NON EXACT (pas de propagation)

	*tmpBig += midh;
	*carry += (*tmpBig < midh);

	*tmpBig += ahbh;
	*carry += (*tmpBig < ahbh);
}

// C = A * B
__device__ void mult(/*reg*/bool posA, /*shared*/uint32_t* decA, /*reg*/bool posB, /*shared*/uint32_t* decB, /*shared*/bool* posC, /*shared*/uint32_t* decC)
{
	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

	uint32_t little = 0; // sera ajoute a decC[k]
	uint32_t big = 0; // sera ajoute a decC[k-1]
	uint8_t carry = 0; // sera ajoute a decC[k-2]

	for (int i = 0; i <= k; i++)
	{
		multDigDig(decA[i], decB[k - i], &carry, &big, &little);
	}

	__syncthreads();

	decC[k] = little;

	__syncthreads();

	if (k > 0)
	{
		decC[k - 1] += big;
		if (decC[k-1] < big)
		{
			carry++;
		}
	}
	else
		*posC = !(posA ^ posB);


	__syncthreads();

	if (k > 1)
		decC[k - 2] += carry; // NON EXACT (pas de propagation)

	__syncthreads();

}

// A = A * B (In Place)
__device__ void multIP(/*shared*/bool* posA, /*shared*/uint32_t* decA, /*shared*/bool* posB, /*shared*/uint32_t* decB)
{
	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

	uint32_t little = 0; // sera ajoute a decA[k]
	uint32_t big = 0; // sera ajoute a decA[k-1]
	uint8_t carry = 0; // sera ajoute a decA[k-2]

	for (int i = 0; i <= k; i++)
	{
		multDigDig(decA[i], decB[k - i], &carry, &big, &little);
	}

	__syncthreads();

	decA[k] = little;

	__syncthreads();

	if (k > 0)
	{
		decA[k - 1] += big;
		if (decA[k - 1] < big)
		{
			carry++;
		}
	}
	else
		*posA = !(*posA ^ *posB);

	__syncthreads();

	if (k > 1)
		decA[k - 2] += carry; // NON EXACT (pas de propagation)

	__syncthreads();
}

__global__ void testKernel(bool* posA, uint32_t* decA, bool* posB, uint32_t* decB, bool* posC, uint32_t* decC)
{
	//const unsigned int ti = threadIdx.x;
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	//__shared__ uint32_t dec[32 * BIG_FLOAT_SIZE];

	if (i == 0)
	{
		// TEST addIP
		addIP(posA, decA, posB, decB);

		// TEST multIP
		//multIP(posA, decA, posB, decB);

		// TEST mult
		//mult(*posA, decA, *posB, decB, posC, decC);
		
		// TEST multDigDig
		/*uint32_t little = 0;
		uint32_t big = 0;
		uint8_t carry = 0;
		
		multDigDig(decA[1], decB[1], &carry, &big, &little);
		decC[0] += carry;
		decC[1] += big;
		decC[2] += little;*/

		// TEST addition
		//add(*posA, decA, *posB, decB, posC, decC);
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

	h_decA[0] = 2;
	h_decA[1] = 6;
	h_decA[2] = 0;
	h_decA[3] = 0;
	//h_decA[1] = 4294967295;
	h_decB[0] = 2;
	h_decB[1] = 9;
	h_decB[2] = 0;
	h_decB[3] = 0;
	//h_decB[1] = 4294967295;
	h_posA = false;
	h_posB = true;

	for (int i = 0; i < BIG_FLOAT_SIZE; i++)
	{
		h_decC[i] = 0;
	}
	h_posC = true;

	ASSERT(cudaSuccess == cudaMemcpy(d_decA, h_decA, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice), "Copy of decA from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_decB, h_decB, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice), "Copy of decB from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_decC, h_decC, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice), "Copy of decC from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_posA, &h_posA, sizeof(bool), cudaMemcpyHostToDevice), "Copy of posA from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_posB, &h_posB, sizeof(bool), cudaMemcpyHostToDevice), "Copy of posB from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_posC, &h_posC, sizeof(bool), cudaMemcpyHostToDevice), "Copy of posC from host to device failed", -1);


	dim3 cudaBlockSize(32, 1, BIG_FLOAT_SIZE); // ATTENTION, 1024 threads max par block
	dim3 cudaGridSize(1, 1, 1);
	testKernel << <cudaGridSize, cudaBlockSize >> >(d_posA, d_decA, d_posB, d_decB, d_posC, d_decC);
	//nothing << <cudaGridSize, cudaBlockSize >> >();

	ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
	ASSERT(cudaSuccess == cudaDeviceSynchronize(), "Kernel synchronization failed", -1);


	ASSERT(cudaSuccess == cudaMemcpy(h_decC, d_decC, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost), "Copy of decC from device to host failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(&h_posC, d_posC, sizeof(bool), cudaMemcpyDeviceToHost), "Copy of posC from device to host failed", -1);

	ASSERT(cudaSuccess == cudaMemcpy(h_decA, d_decA, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost), "Copy of decA from device to host failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(&h_posA, d_posA, sizeof(bool), cudaMemcpyDeviceToHost), "Copy of posA from device to host failed", -1);

	ASSERT(cudaSuccess == cudaMemcpy(h_decB, d_decB, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost), "Copy of decB from device to host failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(&h_posB, d_posB, sizeof(bool), cudaMemcpyDeviceToHost), "Copy of posB from device to host failed", -1);

	ASSERT(cudaSuccess == cudaFree(d_decA), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decB), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decC), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posA), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posB), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posC), "Device deallocation failed", -1);

	cout << "A = " << (h_posA ? "+" : "-") << "  " << h_decA[0] << " " << h_decA[1] << " " << h_decA[2] << " " << h_decA[3] << endl;
	cout << "B = " << (h_posB ? "+" : "-") << "  " << h_decB[0] << " " << h_decB[1] << " " << h_decB[2] << " " << h_decB[3] << endl << endl;
	cout << "C = " << (h_posC ? "+" : "-") << "  " << h_decC[0] << " " << h_decC[1] << " " << h_decC[2] << " " << h_decC[3] << endl;
	

	return EXIT_SUCCESS;
}