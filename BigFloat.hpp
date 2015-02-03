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


	// Fonction servant à multiplier deux chiffres
	static inline bool multDigDig(uint32_t& a, uint32_t& b, uint32_t& first, uint32_t& second, bool carry) {
		uint32_t mask = 0xFFFF;
		uint32_t ah, al, bh, bl, midh, midl, ahbh, ahbl, albh, albl, temp;

		// On se ramene à des int de 16 bits
		ah = a >> 16;
		bh = b >> 16;
		al = mask & a;
		bl = mask & b;

		// Ces produits ne peuvent pas faire d'overflow
		// TODO: certaines mises en mémoire sont inutiles
		ahbh = ah * bh;
		ahbl = ah * bl;
		albh = al * bh;
		albl = al * bl;

		// Le carry est ce qu'on ajoute au chiffre supérieur à la fin

		// On coupe le middle
		midl = ahbl + albh;
		temp = ((ahbl < midl) << 16);
		second += ((ahbl < midl) << 16) + carry; // Ajout de la retenue liée à (ahbl + albh) et du carry issu de la multiplication précédente
		carry = (second < temp + carry); // Initialisation de la prochaine carry
		midh = midl >> 16;
		midl = midl << 16;

		first += albl;
		second += (albl > (first)); // On s’occupe ici de la retenue du chiffre 0 qui concerne donc le chiffre -1
		carry |= ((second) == 0); // Le carry concerne le chiffre d’indice -2 et non pas -1 comme dans l’addition

		first += midl;
		second += (midl > (first));
		carry |= ((second) == 0);

		second += midh;
		carry |= (midh > second);

		second += ahbh;
		carry |= (ahbh > second);

		return carry;
	}


	// Fonction servant à multiplier un chiffre avec la partie entière
	static inline int32_t multDigBase(uint32_t& a, BigFloat& b, uint32_t& first, uint32_t& second, bool carry) {
		uint32_t fakeFirst = 0;
		uint32_t fakeSecond = 0;
		uint32_t ubase;
		if (b.base >= 0) {
			ubase = (b.base);
			carry = multDigDig(a, ubase, first, second, carry);
			return carry;
		}
		else { // Cas très chiant, il faut considérer une retenue négative
			ubase = (-b.base);
			carry = multDigDig(a, ubase, fakeFirst, fakeSecond, carry);
			second -= fakeSecond;
			first -= (second > fakeSecond);
			bool negCarry = (first == 0xFFFFFFFF);

			first -= fakeFirst;
			negCarry |= (first > fakeFirst);
			return -negCarry;
		}
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
		bool carry;

		// Il faut traiter le chiffre de plus basse importance séparément
		uint32_t fakeDigit = 0;
		carry = multDigDig(a[i], b[BIG_FLOAT_SIZE - 1], fakeDigit, temp[BIG_FLOAT_SIZE - 1], 0);

		for (int j = BIG_FLOAT_SIZE - i - 2; j >= 0; j++)
			carry = multDigDig(a[i], b[j], temp[i + j + 1], temp[i + j], carry);

		int32_t baseCarry;
		if (i >= 1) {
			baseCarry = multDigBase(a[i], b, temp[i - 1], temp[i], carry);
			if (i == 1) {
				temp.base += baseCarry;
			}
			// AJOUTER PROPAGATION CARRY
		}
		else {
			uint32_t fakeBase = 0;
			baseCarry = multDigBase(a[i], b, fakeBase, temp[0], carry);
			// ATTENTION A GERER LE SIGNE DE FAKEBASE
		}
	}


	// Multiplication de deux BigFloat. Multiplie b par chaque chiffre de a.
	static void mult(BigFloat& a, BigFloat& b, BigFloat& res) {

	}

};