/* 
 * Classe repr�sentant des flottants sur BIG_FLOAT_SIZE*32 bits
 * Le r�el correspondant est base + dec[0].2^-32 + dec[1].2^-64 + ...
 * base est la partie enti�re du r�el, la partie d�cimale est positive.
 * Dans cette impl�mentation, on fait l�hypoth�se que les overflow sont 
 * impossibles sur la partie enti�re.
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

	BigFloat(BigFloat& a) {
		decimals = new uint32_t[BIG_FLOAT_SIZE];
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = a[i];
		base = a.base;
	}

	~BigFloat() {
		delete[] decimals;
	}

	void reset() {
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = 0;
		base = 0;
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

	// Remplis res avec a+b, res n�a pas besoin d��tre initialis�
	static void add(BigFloat& a, BigFloat& b, BigFloat& res) {
		bool carry = 0;
		for (int i = BIG_FLOAT_SIZE-1; i >= 0; i--) {
			res[i] = a[i] + b[i] + carry;
			carry = (a[i] + 1 == 0 && carry) || ((res[i]) < (a[i] + carry));
		}
		res.base = a.base + b.base + carry;
	}

	// Addition inplace, ajoute a dans b.
	static void add(BigFloat& a, BigFloat& b) {
		bool carry = 0;
		for (int i = BIG_FLOAT_SIZE - 1; i >= 0; i--) {
			b[i] += a[i] + carry;
			carry = (a[i] + 1 == 0 && carry) || ((b[i]) < (a[i] + carry));
		}
		b.base += a.base + carry;
	}

	// N�gation inplace d�un BigFloat
	static void negate(BigFloat& a) {
		// fullZero vaut vrai ssi on n�a rencontr� que des z�ros auparavant, c�est-�-dire qu�il n�y a pas de retenue
		bool fullZero = true;
		for (int i = BIG_FLOAT_SIZE - 1; i >= 0; i--) {
			if (fullZero) {
				a[i] = -a[i];
				fullZero &= (a[i] == 0);
			}
			else
				a[i] = -a[i]-1;
		}
		a.base = - (1-fullZero) - a.base;
	}


	// Fonction servant � multiplier deux chiffres
	static inline bool multDigDig(uint32_t& a, uint32_t& b, uint32_t& little, uint32_t& big, bool carry) {
		uint32_t mask = 0xFFFF;
		uint32_t ah, al, bh, bl, midh, midl, ahbh, ahbl, albh, albl, temp;
		bool tempCarry;

		// On se ramene � des int de 16 bits
		ah = a >> 16;
		bh = b >> 16;
		al = mask & a;
		bl = mask & b;

		// Ces produits ne peuvent pas faire d'overflow
		// TODO: certaines mises en m�moire sont inutiles
		ahbh = ah * bh;
		ahbl = ah * bl;
		albh = al * bh;
		albl = al * bl;

		// Le carry est ce qu'on ajoute au chiffre sup�rieur � la fin

		// On coupe le middle
		midl = ahbl + albh;
		temp = ((midl < ahbl) << 16);
		big += temp + carry; // Ajout de la retenue li�e � (ahbl + albh) et du carry issu de la multiplication pr�c�dente
		carry = (big < temp + carry); // Initialisation de la prochaine carry
		midh = midl >> 16;
		midl = midl << 16;

		little += albl;
		tempCarry = (albl > little);
		big += tempCarry; // On s�occupe ici de la retenue du chiffre 0 qui concerne donc le chiffre -1
		carry |= (((big) == 0) && tempCarry); // Le carry concerne le chiffre d�indice -2 et non pas -1 comme dans l�addition

		little += midl;
		tempCarry =(midl >little);
		big += tempCarry;
		carry |= ((big == 0) && tempCarry);

		big += midh;
		carry |= (big < midh);

		big += ahbh;
		carry |= (big < ahbh);

		return carry;
	}


	// Fonction servant � multiplier un chiffre avec la partie enti�re
	static inline int32_t multDigBase(uint32_t& a, BigFloat& b, uint32_t& little, uint32_t& big, bool carry) {
		uint32_t ubase;
		if (b.base >= 0) {
			ubase = (b.base);
			carry = multDigDig(a, ubase, little, big, carry);
			return carry;
		}
		else { // Cas tr�s chiant, il faut consid�rer une retenue n�gative
			uint32_t fakeLittle = 0;
			uint32_t fakeBig = 0;
			ubase = (-b.base);
			// La carry doit �tre proise en compte manuellement car sinon elle serait retranch�e au lieu d'�tre ajout�e
			multDigDig(a, ubase, fakeLittle, fakeBig, 0);
			little -= fakeLittle;
			bool tempCarry = ((fakeLittle != 0) && (little >= (-fakeLittle)));
			big -= tempCarry;
			int32_t negCarry = (tempCarry && (big == 0xFFFFFFFF)); // Le seul cas d�overflow possible car on a retir� 0 ou 1

			big -= fakeBig;
			negCarry |= ((fakeBig != 0) && (big >= (-fakeBig)));
			// Ajout de la carry issue du calcul pr�c�dent
			big += carry;
			return (carry && (big == 0)) - negCarry;
		}
	}

	static void multOnBase(uint32_t& a, BigFloat& b, BigFloat& temp, bool carry) {
		uint32_t fakeBase = 0; // N�cessaire de passer par un uint32_t
		// La carry renvoy�e vaut 0 ou -1 selon nos hypoth�ses
		// En effet, le d�passement de la base nous am�nerait hors de l�intervalle.
		// Un -1 est possible puisqu�on utilise un uint, il faut alors rendre son signe � fakeBase.
		multDigBase(a, b, temp[0], fakeBase, carry);
		if (b.base >= 0) {
			temp.base += fakeBase;
		}
		else /*if (fakeBase != 0)*/ {
			temp.base -= (-fakeBase); // Si b.base est n�gatif alors fakeBase vaut temp.base + 2^32
		}
	}

	/*  Multiplication de a par un chiffre de b, r�sultat ajout� � temp,
	 *  temp doit �tre initialis� au pr�alable.
	 *  On �vite les integer overflow en utilisant la technique suivante :
	 *	a = ah.2^16 + al ; b = bh.2^16 + bl
	 *  mid = (ahbl + albh) mod 2^32
	 *  carry = ((ahbl + albh) - mid).2^(-32)
	 *	ab = (ahbh).2^32 + mid.2^16 + (albl)
	 *     = (ahbh + carry.2^16 + midh).2^32 + (midl.2^16 + albl)
	 *
	 */
	static void multDigit(BigFloat& a, BigFloat& b, int i, BigFloat& temp) {
		bool carry;

		// Il faut traiter le chiffre de plus basse importance s�par�ment
		uint32_t fakeDigit = 0;
		carry = multDigDig(a[i], b[BIG_FLOAT_SIZE - i - 1], fakeDigit, temp[BIG_FLOAT_SIZE - 1], 0);
		// PRENDRE EN COMPTE UN ARRONDI ?

		// Calcul des multiplications sur les chiffres "standard"
		for (int j = BIG_FLOAT_SIZE - i - 2; j >= 0; j--)
			carry = multDigDig(a[i], b[j], temp[i + j + 1], temp[i + j], carry);


		// Multiplication impliquant la base
		int32_t baseCarry;
		if (i >= 1) {
			// Quand le r�sultat ne va pas dans la base
			baseCarry = multDigBase(a[i], b, temp[i], temp[i - 1], carry); // /!\ Peut �tre n�gative ! /!\
			// Propagation de la retenue
			int k = i - 2;
			while (k >= 0 && (baseCarry != 0)) {
				temp[k] += carry;
				baseCarry = (temp[k] == 0 && baseCarry == 1) - (temp[k] == 0xFFFFFFFF && baseCarry == -1);
				k--;
			}
			// Retenue s�appliquant � la base
			if (k==-1)
				temp.base += baseCarry;
		}
		else {
			// Quand le r�sultat va dans la base
			multOnBase(a[i], b, temp, carry);
		}
	}



	static void multBase(BigFloat& a, BigFloat& b, BigFloat& temp) {
		if (a.base == 0)
			return;
		else if (a.base > 0) {
			bool carry = 0;
			for (int j = BIG_FLOAT_SIZE - 1; j > 0; j--) {
				carry = (multDigBase(b[j], a, temp[j], temp[j - 1], carry) == 1); // Ne peut pas retourner de retenue n�gative puisque la base est positive
			}
		}
		else {
			bool carry;
			for (int j = BIG_FLOAT_SIZE - 1; j > 0; j--) {
				carry = (multDigBase(b[j], a, temp[j], temp[j - 1], 0) == -1); // Retourne une retenue n�gative, on prend l�oppos�
				// Propagation de la retenue n�gative 
				// (pour �viter de remonter � chaque fois il faudrait rendre multDigDig 
				// plus compliqu�e. Cependant, on peut n�gliger ces remont�es de retenues.
				int k = j - 2;
				while (k >= 0 && (carry != 0)) {
					temp[k] -= carry;
					carry = (temp[k] == 0xFFFFFFFF && carry == 1);
					k--;
				}
				if (k == -1)
					temp.base -= carry;
			}
		}
		// Multiplication de la base avec le premier chiffre
		multOnBase(b[0], a, temp, 0);
		// Multiplication base base
		temp.base += a.base * b.base;
	}


	// Multiplication de deux BigFloat.
	// Multiplie b par chaque chiffre de a, ajoute les r�sultats successifs � res.
	// res doit �tre initialis�.
	static void mult(BigFloat& a, BigFloat& b, BigFloat& res) {
		for (int i = BIG_FLOAT_SIZE - 1; i >= 0; i--)
			multDigit(a, b, i, res);
		multBase(a, b, res);
	}

	// Multiplication d�un BigFloat par un float.
	// (utilis� pour les points de d�part)
	// (peut �tre am�lior� en �vitant de repasser par un BigFloat)
	static void mult(float a, BigFloat& b, BigFloat& res) {
		BigFloat aBig(a);
		mult(aBig, b, res);
	}

};