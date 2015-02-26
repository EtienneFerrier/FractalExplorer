/*
Cette classe impl�mente le calcul de l'ensemble de Mandelbrot sur GPU avec precision arbitraire
*/

// Optimisations possibles :
//// Ne pas calculer k (ou i ou j) � chaque fonction
//// Limiter les lectures a la mem globale

#pragma once

#include <cuda.h>
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
// Augmenter divFactor permet d'avoir une fr�quence plus importante de variation des couleurs (mettre 1 par defaut)
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

// Initialise 1/256 en BigFLoat
__device__ void getStep256(bool* pos, uint32_t* dec)
{
	*pos = true;
	dec[0] = 0;
#if BIG_FLOAT_SIZE > 1
	dec[1] = 16766216;
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

// Initialise 1/512 en BigFLoat
__device__ void getStep512(bool* pos, uint32_t* dec)
{
	*pos = true;
	dec[0] = 0;
#if BIG_FLOAT_SIZE > 1
	dec[1] = 16766216/2;
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
			*posA = ((decA[0] == 0) && carry) ^ *posA; // Bottle-neck et NON EXACT (signe donn� par la soustraction des 2 premiers bits)
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
			*posA = ((decA[0] < decB[0]) || ((decA[0] == decB[0]) && carry)) ^ *posA; // Bottle-neck et NON EXACT (signe donn� par la soustraction des 2 premiers bits)
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

//Fonction servant � multiplier deux chiffres
/*	a (reg) et b (reg) sont les copies des decimaux a (shared) et b (shared)
tmpLittle (reg) est un tas qui contient la somme partielle a ajouter plus tard a little (shared)
tmpBig (reg) est un tas qui contient la somme partielle a ajouter plus tard a big (shared)
carry (reg) est un tas qui contient la somme des retenues a ajouter plus tard au decimal avant big (shared)
Chaque appel demande une lecture SHARED -> REG de 2 cases au pr�alable. Possibilit� d'optimiser.*/
__device__ void multDigDig(uint32_t a, uint32_t b, uint8_t* carry, uint32_t* tmpBig, uint32_t* tmpLittle) {

	uint32_t mask = 0xFFFF;
	uint32_t midh, midl, ahbh, ahbl, albh, albl, temp;

	// Ces produits ne peuvent pas faire d'overflow
	// TODO: certaines mises en m�moire sont inutiles
	ahbh = (a >> 16) * (b >> 16);
	ahbl = (a >> 16) * (mask & b);
	albh = (mask & a) * (b >> 16);
	albl = (mask & a) * (mask & b);

	// Le carry est ce qu'on ajoute au chiffre sup�rieur � la fin
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
__device__ void computeMandel(uint32_t* res, int* nbIter, bool* posXinit, uint32_t* decXinit, bool* posYinit, uint32_t* decYinit, bool* posX, uint32_t* decX, bool* posY, uint32_t* decY, bool* posTmp, uint32_t* decTmp, bool* posSq, uint32_t* decSq)
{
	const unsigned int ti = threadIdx.x;
	const unsigned int k = blockDim.z * blockIdx.z + threadIdx.z;

	if (k == 0)
		nbIter[ti] = 0;

	__syncthreads();
	
	// TODO : Verfier la synchronisation (a l'air de marcher)
	/*while (testSquare(posX, decX, posY, decY) && nbIter[ti] < NB_ITERATIONS)
	{
		complexSquare(posX, decX, posY, decY, posTmp, decTmp, posSq, decSq);
		addIP(posX, decX, *posXinit, decXinit);
		addIP(posY, decY, *posYinit, decYinit);

		if (k == 0)
			nbIter[ti] ++;

		__syncthreads();
	}*/
	for (int x = 0; x < NB_ITERATIONS; x++)
	{
		if (testSquare(posX, decX, posY, decY))
		{
			complexSquare(posX, decX, posY, decY, posTmp, decTmp, posSq, decSq);
			addIP(posX, decX, *posXinit, decXinit);
			addIP(posY, decY, *posYinit, decYinit);

			if (k == 0)
				nbIter[ti] ++;
		}

		__syncthreads();
	}

	if (k == 0)
		*res = computeColor_32_DARK(NB_ITERATIONS, nbIter[ti], 2);

	__syncthreads();
}

// Charge le point de depart des iterations dans Res
// Res = (-0.5 + i/WIDTH)*scale + C
// C est le centre de la zone
// x = i ou j
// TODO : voir si passer k en parametre dans toutes les fonctions optimise.
__device__ void loadStart(int n, bool posC, uint32_t* decC, uint32_t* scale, bool* posRes, uint32_t* decRes)
{
	getStep512(posRes, decRes);			// Res = 1/512
	multIntIP(posRes, decRes, true, n);	// Res *= i
	minusHalfIP(posRes, decRes);		// Res -= 0.5
	multIP(posRes, decRes, true, scale);// Res *= scale
	addIP(posRes, decRes, posC, decC);	// Res += C
}

// Fonction principale
__global__ void testKernel(uint32_t* res, bool* posCx, uint32_t* decCx, bool* posCy, uint32_t* decCy, uint32_t* decS)
{
	const unsigned int ti = threadIdx.x;
	const unsigned int i = blockIdx.x*blockDim.x + ti;
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

	__shared__ int nbIter[BLOCK_X];

	loadStart(i, *posCx, decCx, decS, posXinit + ti, decXinit + ti*BIG_FLOAT_SIZE);
	loadStart(j, *posCy, decCy, decS, posYinit + ti, decYinit + ti*BIG_FLOAT_SIZE);

	copyBig(posX + ti, decX + ti*BIG_FLOAT_SIZE, posXinit[ti], decXinit + ti*BIG_FLOAT_SIZE);
	copyBig(posY + ti, decY + ti*BIG_FLOAT_SIZE, posYinit[ti], decYinit + ti*BIG_FLOAT_SIZE);

	computeMandel(res + WIDTH*j + i, nbIter,
		posXinit + ti, decXinit + ti*BIG_FLOAT_SIZE, posYinit + ti, decYinit + ti*BIG_FLOAT_SIZE, 
		posX + ti, decX + ti*BIG_FLOAT_SIZE, posY + ti, decY + ti*BIG_FLOAT_SIZE, 
		posTmp + ti, decTmp + ti*BIG_FLOAT_SIZE, posSq + ti, decSq + ti*BIG_FLOAT_SIZE);
}

// Affiche la fractale de Mandelbrot centree en 0
int computeBigMandelGPU(Affichage* display)
{
	uint32_t* d_res;
	uint32_t* d_decCx;
	uint32_t* d_decCy;
	uint32_t* d_decS;
	bool* d_posCx;
	bool* d_posCy;
	bool* d_posS;

	ASSERT(cudaSuccess == cudaMalloc(&d_res, WIDTH * HEIGHT * sizeof(uint32_t)), "Device allocation of res failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decCx, BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decCx failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decCy, BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decCy failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decS, BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decS failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posCx, sizeof(bool)), "Device allocation of posCx failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posCy, sizeof(bool)), "Device allocation of posCy failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posS, sizeof(bool)), "Device allocation of posS failed", -1);

	uint32_t h_decCx[BIG_FLOAT_SIZE];
	uint32_t h_decCy[BIG_FLOAT_SIZE];
	uint32_t h_decS[BIG_FLOAT_SIZE];
	bool h_posCx;
	bool h_posCy;
	bool h_posS;

	h_decCx[0] = 0;
	h_decCx[1] = 0;// 3006477107;
	h_decCx[2] = 0;
	h_decCx[3] = 0;
	h_posCx = true;

	h_decCy[0] = 0;
	h_decCy[1] = 0;// 2147483648;
	h_decCy[2] = 0;
	h_decCy[3] = 0;
	h_posCy = true;

	h_decS[0] = 4;
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

	dim3 cudaBlockSize(BLOCK_X, BLOCK_Y, BIG_FLOAT_SIZE); // ATTENTION, 1024 threads max par block
	dim3 cudaGridSize(WIDTH / BLOCK_X, HEIGHT / BLOCK_Y, 1); // ATTENTION : changer lorsque la grille n'est pas parfaitement adapt�e a la fenetre
	testKernel << <cudaGridSize, cudaBlockSize >> >(d_res, d_posCx, d_decCx, d_posCy, d_decCy, d_decS);

	ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
	ASSERT(cudaSuccess == cudaDeviceSynchronize(), "Kernel synchronization failed", -1);

	ASSERT(cudaSuccess == cudaMemcpy(display->pixels, d_res, WIDTH * HEIGHT * sizeof(uint32_t), cudaMemcpyDeviceToHost), "Copy of decC from device to host failed", -1);
	
	ASSERT(cudaSuccess == cudaFree(d_res), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decCx), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decCy), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decS), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posCx), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posCy), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posS), "Device deallocation failed", -1);

	return EXIT_SUCCESS;
}

// Affiche la fractale de Mandelbrot avec les dimensions passees en parametre
int computeBigMandelGPU(Affichage* display, bool h_posCx, uint32_t* h_decCx, bool h_posCy, uint32_t* h_decCy, uint32_t* h_decS)
{
	uint32_t* d_res;
	uint32_t* d_decCx;
	uint32_t* d_decCy;
	uint32_t* d_decS;
	bool* d_posCx;
	bool* d_posCy;
	bool* d_posS;

	ASSERT(cudaSuccess == cudaMalloc(&d_res, WIDTH * HEIGHT * sizeof(uint32_t)), "Device allocation of res failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decCx, BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decCx failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decCy, BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decCy failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_decS, BIG_FLOAT_SIZE * sizeof(uint32_t)), "Device allocation of decS failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posCx, sizeof(bool)), "Device allocation of posCx failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posCy, sizeof(bool)), "Device allocation of posCy failed", -1);
	ASSERT(cudaSuccess == cudaMalloc(&d_posS, sizeof(bool)), "Device allocation of posS failed", -1);

	bool h_posS = true;

	ASSERT(cudaSuccess == cudaMemcpy(d_decCx, h_decCx, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice), "Copy of decCx from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_posCx, &h_posCx, sizeof(bool), cudaMemcpyHostToDevice), "Copy of posCx from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_decCy, h_decCy, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice), "Copy of decCy from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_posCy, &h_posCy, sizeof(bool), cudaMemcpyHostToDevice), "Copy of posCy from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_decS, h_decS, BIG_FLOAT_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice), "Copy of decS from host to device failed", -1);
	ASSERT(cudaSuccess == cudaMemcpy(d_posS, &h_posS, sizeof(bool), cudaMemcpyHostToDevice), "Copy of posS from host to device failed", -1);

	dim3 cudaBlockSize(BLOCK_X, BLOCK_Y, BIG_FLOAT_SIZE); // ATTENTION, 1024 threads max par block
	dim3 cudaGridSize(WIDTH / BLOCK_X, HEIGHT / BLOCK_Y, 1); // ATTENTION : changer lorsque la grille n'est pas parfaitement adapt�e a la fenetre
	testKernel << <cudaGridSize, cudaBlockSize >> >(d_res, d_posCx, d_decCx, d_posCy, d_decCy, d_decS);

	ASSERT(cudaSuccess == cudaGetLastError(), "Kernel launch failed", -1);
	ASSERT(cudaSuccess == cudaDeviceSynchronize(), "Kernel synchronization failed", -1);

	ASSERT(cudaSuccess == cudaMemcpy(display->pixels, d_res, WIDTH * HEIGHT * sizeof(uint32_t), cudaMemcpyDeviceToHost), "Copy of decC from device to host failed", -1);

	ASSERT(cudaSuccess == cudaFree(d_res), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decCx), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decCy), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_decS), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posCx), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posCy), "Device deallocation failed", -1);
	ASSERT(cudaSuccess == cudaFree(d_posS), "Device deallocation failed", -1);

	return EXIT_SUCCESS;
}
