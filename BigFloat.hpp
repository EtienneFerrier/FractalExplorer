/* 
 * Classe représentant des flottants sur BIG_FLOAT_SIZE*32 bits
 * Le réel correspondant est base + dec[0].2^-32 + dec[1].2^-64 + ...
 * base est la partie entière du réel, la partie décimale est positive.
 */


#pragma once

#include "Parametres.hpp"
#include <stdint.h>
#include <string>
#include <iostream>
#include <math.h>



class BigFloat {

public:
	uint32_t* decimals; // Chiffres en base 32
	int32_t base; // Chiffre avant la virgule

	BigFloat() {
		decimals = new uint32_t[BIG_FLOAT_SIZE];
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = 0;
		base = 0;
	}

	BigFloat(int k, uint32_t digit) {
		decimals = new uint32_t[BIG_FLOAT_SIZE];
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = 0;
		base = 0;
		decimals[k] = digit;
	}

	BigFloat(float f) {
		decimals = new uint32_t[BIG_FLOAT_SIZE];
		base = (int32_t)floor(f);
		float dec = f - floor(f);

		if (dec == 0.f)
			decimals[0] = 0;
		else
		{
			float logA = 32 * log(2.f) + log(dec);
			decimals[0] = (uint32_t)exp(logA);
		}
		for (int i = 1; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = 0;
	}

	~BigFloat() {
		delete[] decimals;
	}

	void display() {

		int ind = 0; //Permier decimal non nul
		while (ind < BIG_FLOAT_SIZE && decimals[ind] == 0 )
			ind++;

		if (ind == BIG_FLOAT_SIZE)
			std::cout << base << std::endl;
		else
		{
			float logx = log10((float)decimals[ind]) - 32 * (ind + 1)*log10(2.f);
			float n10 = floor(logx);
			float decPart10 = pow(10.f, logx - n10);
			std::cout << base << " + " << decPart10 << ".10^" << (int)n10 << std::endl;
		}
	}

	uint32_t& operator[](int i) {
		return decimals[i];
	}

	// Remplis res avec a+b, res n’a pas besoin d’être initialisé
	static void add(BigFloat& a, BigFloat& b, BigFloat& res) {
		bool carry = 0;
		for (int i = BIG_FLOAT_SIZE-1; i >= 0; i--) {
			res[i] = a[i] + b[i] + carry;
			carry = (a[i] + 1 == 0 && carry) || ((res[i]) < (a[i] + carry));
		}
		res.base = a.base + b.base + carry;
	}

	// Addition inplace
	static void add(BigFloat& a, BigFloat& b) {
		bool carry = 0;
		for (int i = BIG_FLOAT_SIZE - 1; i > 0; i--) {
			b[i] += a[i] + carry;
			carry = (a[i] + 1 == 0 && carry) || ((b[i]) < (a[i] + carry));
		}
		b.base += a.base + carry;
	}

	// Négation inplace d’un BigFloat
	static void negate(BigFloat& a) {
		uint32_t max = -1;
		for (int i = BIG_FLOAT_SIZE - 1; i >= 0; i--) {
			a[i] = max - a[i];
		}
		a.base = -a.base;
	}

	/*  Multiplication de a par un chiffre de b, résultat ajouté à temp,
	 *  temp doit être initialisé au préalable.
	 *  On évite les integer overflow en utilisant la technique suivante :
	 *	a = ah.2^16 + al ; b = bh.2^16 + bl
	 *  mid = (ahbl + albh) mod 2^32
	 *  carry = ((ahbl + albh) - mid).2^(-32)
	 *	ab = (ahbh).2^32 + mid.2^16 + (albl)
	 *     = (ahbh + carry.2^16 + midh).2^32 + (midl.2^16 + albl)
	 *
	 */
	static void multDigit(BigFloat& a, BigFloat& b, int i, BigFloat& temp) {
		bool carry = 0;
		uint32_t mask = 0xFFFF;
		uint32_t ah, al, bh, bl, midh, midl, ahbh, ahbl, albh, albl;

		// Il faut traiter le chiffre de plus basse importance séparément

		uint32_t fakeDigit = 0;
		// On se ramene à des int de 16 bits
		ah = a[i] >> 16;
		bh = b[i] >> 16;
		al = mask & a[i];
		bl = mask & b[i];

		// Ces produits ne peuvent pas faire d'overflow
		// TODO: certaines mises en mémoire sont inutiles
		ahbh = ah * bh;
		ahbl = ah * bl;
		albh = al * bh;
		albl = al * bl;

		// Le carry est ce qu'on ajoute au chiffre supérieur à la fin

		// On coupe le middle
		midl = ahbl + albh;
		temp[BIG_FLOAT_SIZE-1] += ((ahbl < midl) << 16);
		midh = midl >> 16;
		midl = midl << 16;

		fakeDigit += albl;
		temp[BIG_FLOAT_SIZE] += (albl > fakeDigit); // On s’occupe ici de la retenue du chiffre 0 qui concerne donc le chiffre -1
		carry = ((temp[BIG_FLOAT_SIZE-1]) == 0); // Le carry concerne le chiffre d’indice -2 et non pas -1 comme dans l’addition
		fakeDigit += midl + carry;
		carry |= (midl + 1 == 0 && carry) || (midl + carry > fakeDigit);

		for (int j = BIG_FLOAT_SIZE-i-2; j >= 0; j++) {

			// On se ramene à des int de 16 bits
			ah = a[i] >> 16;
			bh = b[i] >> 16;
			al = mask & a[i];
			bl = mask & b[i];

			// Ces produits ne peuvent pas faire d'overflow
			// TODO: certaines mises en mémoire sont inutiles
			ahbh = ah * bh;
			ahbl = ah * bl;
			albh = al * bh;
			albl = al * bl;

			// Le carry est ce qu'on ajoute au chiffre supérieur à la fin

			// On coupe le middle
			midl = ahbl + albh;
			temp[i+j-2] += ((ahbl < midl) << 16);
			midh = midl >> 16;
			midl = midl << 16;

			temp[i + j - 1] += albl;
			temp[i + j - 2] += (albl > temp[i + j - 1]); // On s’occupe ici de la retenue du chiffre 0 qui concerne donc le chiffre -1
			carry = ((temp[i + j - 2]) == 0); // Le carry concerne le chiffre d’indice -2 et non pas -1 comme dans l’addition
			temp[i + j - 1] += midl + carry;
			carry |= (midl + 1 == 0 && carry) || (midl + carry > temp[i + j - 1]);
		}


	}
	
	static void mult(BigFloat& a, BigFloat& b, BigFloat& res) {
		BigFloat temp;
		BigFloat digit;
		for (int i = 0; i < BIG_FLOAT_SIZE; i++) {
			digit = BigFloat(i)
		}
	}

};