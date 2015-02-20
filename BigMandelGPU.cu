/*
Cette classe implémente le calcul de l'ensemble de Mandelbrot sur GPU.
*/

// Optimisations possibles
// Ne pas calculer k à chaque fonction

#pragma once

#include <cuda.h>
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SDL2/SDL.h>

#include "Parametres.hpp"
#include "Affichage.hpp"
#include "BigFloat2.hpp"



#define ASSERT(x, msg, retcode) \
    if (!(x)) \
			    { \
        cout << msg << " " << __FILE__ << ":" << __LINE__ << endl; \
        return retcode; \
			    }

// Affiche en float le BigFloat en prenant en compte les 2 premiers decimaux seulement
void display2dec(bool pos, uint32_t* dec)
{
	double logx;
	double n10;
	double decPart10;

	if (dec[1] != 0)
	{
		logx = log10((double)dec[1]) - 32.*log10(2.);
		n10 = floor(logx);
		decPart10 = pow(10.f, logx - n10);
	}
	else
	{
		decPart10 = 0;
		n10 = 0;
	}
	cout << (pos ? "+" : "-") << dec[0] << " + " << decPart10 << ".10^" << (int)n10 << endl;
}

// Initialise 1/100 en BigFLoat
// Pourrait etre une macro preproc
// A changer des que l'on change la taille de la fenetre
__device__ void getStep100(bool* pos, uint32_t* dec)
{
	*pos = true;
	dec[0] = 0;
#if BIG_FLOAT_SIZE > 1
	dec[1] = 42949673;
#endif
#if BIG_FLOAT_SIZE > 2
	dec[2] = 0;
#endif
#if BIG_FLOAT_SIZE > 3
	dec[3] = 0;
#endif
#if BIG_FLOAT_SIZE > 4
	dec[4] = 0;
#endif
#if BIG_FLOAT_SIZE > 4
	dec[5] = 0;
#endif
#if BIG_FLOAT_SIZE > 4
	dec[6] = 0;
#endif

	__syncthreads();
}

// Initialise 1/600 en BigFLoat
// Pourrait etre une macro preproc
// A changer des que l'on change la taille de la fenetre
__device__ void getStep600(bool* pos, uint32_t* dec)
{
	*pos = true;
	dec[0] = 0;
#if BIG_FLOAT_SIZE > 1
	dec[1] = 7158279;
#endif
#if BIG_FLOAT_SIZE > 2
	dec[2] = 0;
#endif
#if BIG_FLOAT_SIZE > 3
	dec[3] = 0;
#endif
#if BIG_FLOAT_SIZE > 4
	dec[4] = 0;
#endif
#if BIG_FLOAT_SIZE > 4
	dec[5] = 0;
#endif
#if BIG_FLOAT_SIZE > 4
	dec[6] = 0;
#endif
}

// Enleve 0.5 a A 
// (0.5 = 0 0x80000000 0 0 0 0 ...)
// Meme fonctionnement que addIP
__device__ void minusHalfIP(bool* posA, uint32_t* decA)
{
	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

	uint32_t tmp; // Sera ajoute a posA[k]
	bool carry; // Sera ajoute a posA[k-1]

	if (*posA == false)	// Cas simple (addition unsigned)
	{
		if (k == 1)
			tmp = decA[k] + 0x80000000;
		else
			tmp = decA[k];

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
		if (k == 1)
			tmp = decA[k] - 0x80000000; // Soustraction
		else
			tmp = decA[k];

		carry = tmp > decA[k]; // Calcul des retenues 

		if (k == 1)
		{
			*posA = ((decA[0] == 0) && carry) ^ *posA; // Bottle-neck et NON EXACT (signe donné par la soustraction des 2 premiers bits)
		}
		__syncthreads();

		if (*posA == false) // Dans ce cas, |B| > |A|. On doit donc recalculer la soustraction
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

// Change le signe
__device__ void negate(bool* pos)
{
	*pos = !*pos;
}

// C = A + B
//__device__ void add(bool posA, uint32_t* decA, bool posB, uint32_t* decB, bool* posC, uint32_t* decC)
//{
//	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;
//	bool carry;
//
//	if (posA == posB)	// Cas simple (addition unsigned)
//	{
//		decC[k] = decA[k] + decB[k]; // Addition
//		carry = decC[k] < decA[k]; // Calcul des retenues
//		__syncthreads();
//
//		if (k > 0) 
//		{
//			decC[k - 1] += carry; // Propagation des retenues (une seule fois)
//		}
//		else
//		{
//			*posC = posA; // Calcul du signe du resultat
//		}
//		__syncthreads();
//	}
//	else // Cas complique (soustraction)
//	{
//		decC[k] = decA[k] - decB[k]; // Soustraction
//		carry = decC[k] > decA[k]; // Calcul des retenues 
//		__syncthreads();
//
//		if (k > 0)
//		{
//			decC[k-1] -= carry; // Propagation des retenues (une seule fois)
//		}
//		__syncthreads();
//
//		if(k == 0)
//		{
//			*posC = (carry || decC[0] == -1) ^ posA; // ATTENTION : pas le signe de C mais un indicateur de |A| < |B| (valeur absolue)
//		}
//		__syncthreads();
//
//		if (*posC ^ posA) // Dans ce cas, |B| > |A|. On doit donc recalculer la soustraction
//		{
//			decC[k] = decB[k] - decA[k];
//			carry = decC[k] > decB[k];
//			__syncthreads();
//
//			if (k > 0)
//			{
//				decC[k - 1] -= carry;
//			}
//			else
//			{
//				*posC = posB; // Signe du resultat
//			}
//			__syncthreads();
//		}
//	}
//}

// A = A + B (In Place)
__device__ void addIP(bool* posA, uint32_t* decA, bool *posB, uint32_t* decB)
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
__device__ void multDigDig(uint32_t a, uint32_t b, uint8_t* carry, uint32_t* tmpBig, uint32_t* tmpLittle) {

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
//__device__ void mult(bool posA, uint32_t* decA, bool posB, uint32_t* decB, bool* posC, uint32_t* decC)
//{
//	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;
//
//	uint32_t little = 0; // sera ajoute a decC[k]
//	uint32_t big = 0; // sera ajoute a decC[k-1]
//	uint8_t carry = 0; // sera ajoute a decC[k-2]
//
//	for (int i = 0; i <= k; i++)
//	{
//		multDigDig(decA[i], decB[k - i], &carry, &big, &little);
//	}
//
//	__syncthreads();
//
//	decC[k] = little;
//
//	__syncthreads();
//
//	if (k > 0)
//	{
//		decC[k - 1] += big;
//		if (decC[k-1] < big)
//		{
//			carry++;
//		}
//	}
//	else
//		*posC = !(posA ^ posB);
//
//
//	__syncthreads();
//
//	if (k > 1)
//		decC[k - 2] += carry; // NON EXACT (pas de propagation)
//
//	__syncthreads();
//
//}

// A = A * B (In Place)
__device__ void multIP(bool* posA, uint32_t* decA, bool* posB, uint32_t* decB)
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

// A = factor * A
__device__ void multIntIP(bool* posA, uint32_t* decA, bool posFactor, uint32_t factor)
{
	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

	uint32_t little = 0; // sera ajoute a decC[k]
	uint32_t big = 0; // sera ajoute a decC[k-1]
	uint8_t carry = 0; // sera ajoute a decC[k-2]

	multDigDig(decA[k], factor, &carry, &big, &little);

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
		*posA = !(*posA ^ posFactor);


	__syncthreads();

	if (k > 1)
		decA[k - 2] += carry; // NON EXACT (pas de propagation)

	__syncthreads();
}

// Charge le point de depart dans Res
// C est le centre de la zone
// TODO : passer i en parametre pour avoir une seule fonction pour i et j.
// TODO : voir si passer k en parametre dans toutes les fonctions optimise.
__device__ void loadStart(bool posC, uint32_t* decC, uint32_t* scale, bool* posRes, uint32_t* decRes)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	//unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

	bool posScale = true;

	getStep100(posRes, decRes);
	multIntIP(posRes, decRes, true, i);
	minusHalfIP(posRes, decRes);
	/*multIP(posRes, decRes, &posScale, scale);
	addIP(posRes, decRes, &posC, decC);*/
}


__global__ void testKernel(bool* posA, uint32_t* decA, bool* posB, uint32_t* decB, bool* posC, uint32_t* decC)
{
	//const unsigned int ti = threadIdx.x;
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	//__shared__ uint32_t dec[32 * BIG_FLOAT_SIZE];

	if (i == 35)
	{
		// Test loadStart
		loadStart(*decB, decB, decB, posA, decA);

		// Test minusHalfIP
		//minusHalfIP(posA, decA);

		// Test multIntIP
		//multIntIP(posA, decA, false, -1);

		// TEST addIP
		//addIP(posA, decA, posB, decB);

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

	h_decA[0] = 0;
	h_decA[1] = 0x80000000;
	h_decA[2] = 0;
	h_decA[3] = 0;

	h_decB[0] = 2;
	h_decB[1] = 7;
	h_decB[2] = 3;
	h_decB[3] = 0;

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

	cout << " A =  " << endl; 
	display2dec(h_posA, h_decA);
	
	/*BigFloat2 xStep(1. / 600.);
	cout << xStep.decimals[0] << "   " << xStep.decimals[1] << "   " << xStep.decimals[2] << "   " << xStep.decimals[3] << endl;*/

	return EXIT_SUCCESS;
}