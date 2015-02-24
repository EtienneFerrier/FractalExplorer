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



#define ASSERT(x, msg, retcode) \
    if (!(x)) \
				    { \
        cout << msg << " " << __FILE__ << ":" << __LINE__ << endl; \
        return retcode; \
				    }

// Affiche en float le BigFloat en utilisant les 2 premieres decimales uniquement
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

// Affiche un tableau de BigFloat de maniere brute
void displayBigArray(bool* pos, uint32_t* dec, int arraySize)
{
	for (int i = 0; i < arraySize; i++)
	{
		cout << "(" << i << ")  :  " << (pos[i] ? "+" : "-") << "  " << dec[BIG_FLOAT_SIZE * i] << " " << dec[BIG_FLOAT_SIZE * i + 1] << " " << dec[BIG_FLOAT_SIZE * i + 2] << endl;
	}
}

// Affiche en float un tableau de BigFLoat en utilisant les 2 premieres decimales uniquement
void displayBigArray2dec(bool* pos, uint32_t* dec, int arraySize)
{
	for (int i = 0; i < arraySize; i++)
	{
		display2dec(pos[i], dec + i*BIG_FLOAT_SIZE);
	}
}

// Convertit RGB en Uint32
__device__ uint32_t couleur(int r, int g, int b)
{
	return (((r << 8) + g) << 8) + b;
}

// Methode de coloration en couleur 32bits version sombre
// Augmenter divFactor permet d'avoir une fréquence plus importante de variation des couleurs (mettre 1 par defaut)
__device__ uint32_t computeColor_32_DARK(int countMax, int count, int divFactor)
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
	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

	if (k == 0)
		*pos = !*pos;

	__syncthreads();
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
__device__ void addIP(bool* posA, uint32_t* decA, bool posB, uint32_t* decB)
{
	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

	uint32_t tmp; // Sera ajoute a posA[k]
	bool carry; // Sera ajoute a posA[k-1]

	if (*posA == posB)	// Cas simple (addition unsigned)
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

		if (*posA == posB) // Dans ce cas, |B| > |A|. On doit donc recalculer la soustraction
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
__device__ void multIP(bool* posA, uint32_t* decA, bool posB, uint32_t* decB)
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
		*posA = !(*posA ^ posB);

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

// A = B
__device__ void copyBig(bool* posA, uint32_t* decA, bool posB, uint32_t* decB)
{
	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

	if (k == 0)
	{
		*posA = posB;
	}
	decA[k] = decB[k];

	__syncthreads();
}

// X = X*X - Y*Y
// Y = 2*X*Y
__device__ void complexSquare(bool* posX, uint32_t* decX, bool* posY, uint32_t* decY, bool* posTmp, uint32_t* decTmp, bool* posSq, uint32_t* decSq)
{
	// tmp = 2*X*Y
	copyBig(posTmp, decTmp, *posX, decX);	// tmp = X
	multIP(posTmp, decTmp, *posY, decY);	// tmp *= Y
	multIntIP(posTmp, decTmp, true, 2);		// tmp *= 2

	// X = X*X - Y*Y
	// Sq = - Y * Y
	multIP(posX, decX, *posX, decX);		// X *= X
	copyBig(posSq, decSq, *posY, decY);		// Sq = Y
	multIP(posSq, decSq, *posY, decY);		// Sq *= Y
	negate(posSq);							// neg(Sq)
	addIP(posX, decX, *posSq, decSq);		// X += Sq

	// Y = tmp
	copyBig(posY, decY, *posTmp, decTmp);

}

// Realise le test X^2 + Y^2 < 4 en place.
// Ne modifie pas les valeurs de X et Y.
__device__ bool testSquare(bool* posX, uint32_t* decX, bool* posY, uint32_t* decY)
{
	const unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

	uint32_t recDecX = decX[k];
	uint32_t recDecY = decY[k];
	bool recPosX = *posX;
	bool recPosY = *posY;
	
	bool res;

	multIP(posX, decX, *posX, decX); // X *= X
	multIP(posY, decY, *posY, decY); // Y *= Y
	addIP(posX, decX, *posY, decY); // X^2 += Y^2

	if (decX[0] < 4)
		res = true;
	else
		res = false;

	__syncthreads();

	*posX = recPosX;
	*posY = recPosY;

	decX[k] = recDecX;
	decY[k] = recDecY;

	__syncthreads();

	return res;
}

// Boucle d'iteration principale
// TODO : verifier la synchronisation (a l'air de marcher)
__device__ void computeMandel(uint32_t* res, bool* posXinit, uint32_t* decXinit, bool* posYinit, uint32_t* decYinit, bool* posX, uint32_t* decX, bool* posY, uint32_t* decY, bool* posTmp, uint32_t* decTmp, bool* posSq, uint32_t* decSq)
{
	const unsigned int ti = threadIdx.x;
	const unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	__shared__ int nbIter[BLOCK_X];

	if (k == 0)
		nbIter[ti] = 0;

	__syncthreads();
	
	// TODO : Verfier la synchronisation (a l'air de marcher)
	while (testSquare(posX, decX, posY, decY) && nbIter[ti] < NB_ITERATIONS)
	{
		complexSquare(posX, decX, posY, decY, posTmp, decTmp, posSq, decSq);
		addIP(posX, decX, *posXinit, decXinit);
		addIP(posY, decY, *posYinit, decYinit);

		if (k == 0)
			nbIter[ti] ++;

		__syncthreads();
	}

	if (k == 0)
		res[i] = computeColor_32_DARK(NB_ITERATIONS, nbIter[ti], 1);
		//res[i] = nbIter[ti];
	

	__syncthreads();
}

// Charge le point de depart des iterations dans Res
// Res = (-0.5 + i/WIDTH)*scale + C
// C est le centre de la zone
// x = i ou j
// TODO : voir si passer k en parametre dans toutes les fonctions optimise.
__device__ void loadStart(int x, bool posC, uint32_t* decC, uint32_t* scale, bool* posRes, uint32_t* decRes)
{
	getStep100(posRes, decRes);			// Res = 1/100
	multIntIP(posRes, decRes, true, x);	// Res *= i
	minusHalfIP(posRes, decRes);		// Res -= 0.5
	multIP(posRes, decRes, true, scale);// Res *= scale
	addIP(posRes, decRes, posC, decC);	// Res += C
}

__global__ void testKernel(uint32_t* res, bool* posXref, uint32_t* decXref, bool* posYref, uint32_t* decYref, bool* posCx, uint32_t* decCx, bool* posCy, uint32_t* decCy, uint32_t* decS)
{
	const unsigned int ti = threadIdx.x;
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

	__shared__ uint32_t decXinit[BLOCK_X * BIG_FLOAT_SIZE];
	__shared__ uint32_t decYinit[BLOCK_X * BIG_FLOAT_SIZE];
	__shared__ uint32_t decX[BLOCK_X * BIG_FLOAT_SIZE];
	__shared__ uint32_t decY[BLOCK_X * BIG_FLOAT_SIZE];
	__shared__ uint32_t decTmp[BLOCK_X * BIG_FLOAT_SIZE];
	__shared__ uint32_t decSq[BLOCK_X * BIG_FLOAT_SIZE];

	__shared__ bool posXinit[BLOCK_X];
	__shared__ bool posYinit[BLOCK_X];
	__shared__ bool posX[BLOCK_X];
	__shared__ bool posY[BLOCK_X];
	__shared__ bool posTmp[BLOCK_X];
	__shared__ bool posSq[BLOCK_X];

	// Test iterate
	loadStart(i, *posCx, decCx, decS, posXinit + ti, decXinit + ti*BIG_FLOAT_SIZE);
	loadStart(j, *posCy, decCy, decS, posYinit + ti, decYinit + ti*BIG_FLOAT_SIZE);

	copyBig(posX + ti, decX + ti*BIG_FLOAT_SIZE, posXinit[ti], decXinit + ti*BIG_FLOAT_SIZE);
	copyBig(posY + ti, decY + ti*BIG_FLOAT_SIZE, posYinit[ti], decYinit + ti*BIG_FLOAT_SIZE);

	computeMandel(res, 
		posXinit + ti, decXinit + ti*BIG_FLOAT_SIZE, posYinit + ti, decYinit + ti*BIG_FLOAT_SIZE, 
		posX + ti, decX + ti*BIG_FLOAT_SIZE, posY + ti, decY + ti*BIG_FLOAT_SIZE, 
		posTmp + ti, decTmp + ti*BIG_FLOAT_SIZE, posSq + ti, decSq + ti*BIG_FLOAT_SIZE);

	//res[i] = testSquare(posX + ti, decX + ti*BIG_FLOAT_SIZE, posY + ti, decY + ti*BIG_FLOAT_SIZE);
	
	
	copyBig(posXref + ti, decXref + ti*BIG_FLOAT_SIZE, posX[ti], decX + ti*BIG_FLOAT_SIZE);
	copyBig(posYref + ti, decYref + ti*BIG_FLOAT_SIZE, posY[ti], decY + ti*BIG_FLOAT_SIZE);
	
	/*
	// Test testSquare
	//posY[i] = testSquare(posX + i, decX + i*BIG_FLOAT_SIZE, posY + i, decY + i*BIG_FLOAT_SIZE);

	// Test loadStart multiple
	//loadStart(i, *posC, decC, decS, posX + i, decX + i*BIG_FLOAT_SIZE);
	//loadStart(j, *posC, decC, decS, posY + i, decY + i*BIG_FLOAT_SIZE); // Changer cet appel quand intégration de la dimension j

	
	if (i == 2)
	{
		// Test complexSquare
		//complexSquare(posA, decA, posB, decB, &posTmp, decTmp, &posSq, decSq);

		// Test squareIP
		//multIP(posB, decB, *posA, decA);
		//multIP(posA, decA, *posA, decA);

		// Test negate
		//negate(posA);

		// Test loadStart
		//loadStart(*decA, decA, decB, posC, decC);

		// Test minusHalfIP
		//minusHalfIP(posA, decA);

		// Test multIntIP
		//multIntIP(posA, decA, false, -1);

		// TEST addIP
		//addIP(posA, decA, *posB, decB);

		// TEST multIP
		//multIP(posA, decA, *posB, decB);

		// TEST mult
		//mult(*posA, decA, *posB, decB, posC, decC);
		
		// TEST multDigDig
		//uint32_t little = 0;
		//uint32_t big = 0;
		//uint8_t carry = 0;
		//
		//multDigDig(decA[1], decB[1], &carry, &big, &little);
		//decC[0] += carry;
		//decC[1] += big;
		//decC[2] += little;

		// TEST addition
		//add(*posA, decA, *posB, decB, posC, decC);
	}*/

}

// Fonction de communication avec le GPU, lance les thread et gère les échanges mémoire
int testBigMandelGPU()
{
	uint32_t* d_res;
	uint32_t* d_decX;
	uint32_t* d_decY;
	uint32_t* d_decCx;
	uint32_t* d_decCy;
	uint32_t* d_decS;
	bool* d_posX;
	bool* d_posY;
	bool* d_posCx;
	bool* d_posCy;
	bool* d_posS;

	ASSERT(cudaSuccess == cudaMalloc(&d_res, BLOCK_X * sizeof(uint32_t)), "Device allocation of res failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decX, BLOCK_X * BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decX failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decY, BLOCK_X * BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decY failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decCx, BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decCx failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decCy, BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decCy failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decS, BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decS failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posX, BLOCK_X * sizeof(bool)), "Device allocation of posX failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posY, BLOCK_X * sizeof(bool)), "Device allocation of posY failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posCx, sizeof(bool)), "Device allocation of posC failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posCy, sizeof(bool)), "Device allocation of posC failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posS, sizeof(bool)), "Device allocation of posS failed", -1);

	uint32_t h_res[BLOCK_X];
	uint32_t h_decX[BLOCK_X * BIG_FLOAT_SIZE];
	uint32_t h_decY[BLOCK_X * BIG_FLOAT_SIZE];
	uint32_t h_decCx[BIG_FLOAT_SIZE];
	uint32_t h_decCy[BIG_FLOAT_SIZE];
	uint32_t h_decS[BIG_FLOAT_SIZE];
	bool h_posX[BLOCK_X];
	bool h_posY[BLOCK_X];
	bool h_posCx;
	bool h_posCy;
	bool h_posS;

	h_decCx[0] = 0;
	h_decCx[1] = 3006477107;
	h_decCx[2] = 0;
	h_decCx[3] = 0;
	h_posCx = true;

	h_decCy[0] = 0;
	h_decCy[1] = 0;
	h_decCy[2] = 0;
	h_decCy[3] = 0;
	h_posCy = true;

	h_decS[0] = 1;
	h_decS[1] = 0;
	h_decS[2] = 0;
	h_decS[3] = 0;
	h_posS = true;

	ASSERT(cudaSuccess == cudaMemcpy(d_decCx, h_decCx, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice), "Copy of decCx from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_posCx, &h_posCx, sizeof(bool), cudaMemcpyHostToDevice), "Copy of posCx from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_decCy, h_decCy, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice), "Copy of decCy from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_posCy, &h_posCy, sizeof(bool), cudaMemcpyHostToDevice), "Copy of posCy from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_decS, h_decS, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice), "Copy of decS from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_posS, &h_posS, sizeof(bool), cudaMemcpyHostToDevice), "Copy of posS from host to device failed", -1);

	dim3 cudaBlockSize(BLOCK_X, 1, BIG_FLOAT_SIZE); // ATTENTION, 1024 threads max par block
	dim3 cudaGridSize(1, 1, 1);
	testKernel << <cudaGridSize, cudaBlockSize >> >(d_res, d_posX, d_decX, d_posY, d_decY, d_posCx, d_decCx, d_posCy, d_decCy, d_decS);

	ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
	ASSERT(cudaSuccess == cudaDeviceSynchronize(), "Kernel synchronization failed", -1);

	ASSERT(cudaSuccess == cudaMemcpy(h_res, d_res, BLOCK_X * sizeof(uint32_t), cudaMemcpyDeviceToHost), "Copy of decC from device to host failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(h_decX, d_decX, BLOCK_X * BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost), "Copy of decX from device to host failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(&h_posX, d_posX, BLOCK_X * sizeof(bool), cudaMemcpyDeviceToHost), "Copy of posX from device to host failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(h_decY, d_decY, BLOCK_X * BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost), "Copy of decY from device to host failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(&h_posY, d_posY, BLOCK_X * sizeof(bool), cudaMemcpyDeviceToHost), "Copy of posY from device to host failed", -1);

	ASSERT(cudaSuccess == cudaFree(d_res), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decX), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decY), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decCx), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decCy), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decS), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posX), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posY), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posCx), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posCy), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posS), "Device deallocation failed", -1);
	
	displayBigArray2dec(h_posX, h_decX, BLOCK_X);
	cout << "========" << endl;
	displayBigArray2dec(h_posY, h_decY, BLOCK_X);
	cout << "========" << endl;
	for (int i = 0; i < BLOCK_X; i++)
	{
		cout << h_res[i] << endl;
	}


	return EXIT_SUCCESS;
}