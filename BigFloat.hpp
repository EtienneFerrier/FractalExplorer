/*
* Classe représentant des flottants sur BIG_FLOAT_SIZE*32 bits
* Le réel correspondant est (2*pos-1)(dec[0] + dec[1].2^-32 + ... )
* pos vaut est un booléen valant true si le nombre est positif.
* Dans cette implémentation, on fait l’hypothèse que les overflow sont
* impossibles sur le premier chiffre.
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
	bool pos; // Vrai si positif, faux si négatif

	BigFloat() {
		decimals = new uint32_t[BIG_FLOAT_SIZE];
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = 0;
		pos = 1;
	}

	BigFloat(int k, uint32_t digit) {
		decimals = new uint32_t[BIG_FLOAT_SIZE];
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = 0;
		pos = 1;
		decimals[k] = digit;
	}

	BigFloat(double d) {
		decimals = new uint32_t[BIG_FLOAT_SIZE];
		pos = d>=0;
		if (!pos)
			d = -d;
		uint64_t base = 0x100000000;
		for (int i = 0; i < BIG_FLOAT_SIZE; i++) {
			decimals[i] = floor(d);
			d -= decimals[i];
			d *= base;
		}
	}

	// Prends des paquets de quatre chiffres
	// Exemple :  BigFloat(true, 0, 3750, 0012, 0061, 8655, 0)
	// Ne fonctionne pas très bien àcause des arrondis du log
	BigFloat(bool pos, float b, float c, float d, float e, float f, float g) {
		decimals = new uint32_t[BIG_FLOAT_SIZE];
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = 0;
		this->pos = pos;
		BigFloat bBig(b);
		mult(1, bBig, *this);
		BigFloat cBig(c);
		mult(1e-4f, cBig, *this);
		BigFloat dBig(d);
		mult(1e-8f, dBig, *this);
		BigFloat eBig(e);
		mult(1e-12f, eBig, *this);
		BigFloat fBig(f);
		mult(1e-16f, fBig, *this);
		BigFloat gBig(g);
		mult(1e-20f, gBig, *this);
	}

	// Directement des chiffres
	// Exemple : BigFloat(false, 1, 3178543730, 764955228, 0)
	BigFloat(int32_t pos, uint32_t b, uint32_t c, uint32_t d, uint32_t e) {
		decimals = new uint32_t[BIG_FLOAT_SIZE];
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = 0;
		if (b != 0)
			decimals[0] = b;
		if (c != 0)
			decimals[1] = c;
		if (d != 0)
			decimals[2] = d;
		if (e != 0)
			decimals[3] = e;
		this->pos = pos;
	}


	BigFloat(BigFloat& a) {
		decimals = new uint32_t[BIG_FLOAT_SIZE];
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = a[i];
		pos = a.pos;
	}

	~BigFloat() {
		delete[] decimals;
	}

	void reset() {
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = 0;
		pos = true;
	}

	void copy(BigFloat a) {
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = a.decimals[i];
		pos = a.pos;
	}


	uint32_t& operator[](int i) {
		return decimals[i];
	}

	// Remplis res avec a+b, res n’a pas besoin d’être initialisé
	static void add(BigFloat& a, BigFloat& b, BigFloat& res) {
		bool carry = 0;

		if (a.pos == b.pos) {
			for (int i = BIG_FLOAT_SIZE - 1; i >= 0; i--) {
				res[i] = a[i] + b[i] + carry;
				carry = (a[i] + 1 == 0 && carry) || ((res[i]) < (a[i] + carry));
			}
			res.pos = a.pos;
		}
		else {
			int j = 0;
			while (j < BIG_FLOAT_SIZE && a[j] == b[j])
				j++;
			bool aBigger = a[j] >= b[j];

			if (aBigger) {
				for (int i = BIG_FLOAT_SIZE - 1; i >= 0; i--) {
					res[i] = a[i] - carry;
					bool nextCarry = (a[i] + 1 == 0 && carry);
					res[i] -= b[i];
					carry = nextCarry || ((res[i] >= -b[i]) && (b[i] != 0));
				}
				res.pos =  a.pos;
			} 
			else {
				for (int i = BIG_FLOAT_SIZE - 1; i >= 0; i--) {
					res[i] = b[i] - carry;
					bool nextCarry = (b[i] + 1 == 0 && carry);
					res[i] -= a[i];
					carry = nextCarry || ((res[i] >= -a[i]) && (a[i] != 0));
				}
				res.pos = b.pos;
			}
		}
	}


	// Négation inplace d’un BigFloat
	static void negate(BigFloat& a) {
		a.pos = !a.pos;
	}


	// Fonction servant à multiplier deux chiffres
	// On évite les integer overflow en utilisant la technique suivante :
	// a = ah.2 ^ 16 + al; b = bh.2 ^ 16 + bl
	// mid = (ahbl + albh) mod 2 ^ 32
	// carry = ((ahbl + albh) - mid).2 ^ (-32)
	// ab = (ahbh).2 ^ 32 + mid.2 ^ 16 + (albl)
	//	  = (ahbh + carry.2 ^ 16 + midh).2 ^ 32 + (midl.2 ^ 16 + albl)
	static inline bool multDigDig(uint32_t& a, uint32_t& b, uint32_t& little, uint32_t& big, bool carry) {
		uint32_t mask = 0xFFFF;
		uint32_t ah, al, bh, bl, midh, midl, ahbh, ahbl, albh, albl, temp;
		bool tempCarry;

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
		temp = ((midl < ahbl) << 16);
		big += temp + carry; // Ajout de la retenue liée à (ahbl + albh) et du carry issu de la multiplication précédente
		carry = (big < temp + carry); // Initialisation de la prochaine carry
		midh = midl >> 16;
		midl = midl << 16;

		little += albl;
		tempCarry = (albl > little);
		big += tempCarry; // On s’occupe ici de la retenue du chiffre 0 qui concerne donc le chiffre -1
		carry |= (((big) == 0) && tempCarry); // Le carry concerne le chiffre d’indice -2 et non pas -1 comme dans l’addition

		little += midl;
		tempCarry = (midl >little);
		big += tempCarry;
		carry |= ((big == 0) && tempCarry);

		big += midh;
		carry |= (big < midh);

		big += ahbh;
		carry |= (big < ahbh);

		return carry;
	}



	// Multiplication de a par un chiffre de b, résultat ajouté à temp,
	// temp doit être initialisé au préalable.
	static void multDigit(BigFloat& a, BigFloat& b, int i, BigFloat& temp) {
		bool carry;

		// Il faut traiter le chiffre de plus basse importance séparément
		uint32_t fakeDigit = 0;
		carry = multDigDig(a[i], b[BIG_FLOAT_SIZE - i], fakeDigit, temp[BIG_FLOAT_SIZE - 1], 0);
		// PRENDRE EN COMPTE UN ARRONDI ?

		// Calcul des multiplications sur les chiffres "standard"
		for (int j = BIG_FLOAT_SIZE - i - 1; j >= 0; j--)
			carry = multDigDig(a[i], b[j], temp[i + j], temp[i + j - 1], carry);

	}


	// Multiplication de deux BigFloat.
	// Multiplie b par chaque chiffre de a, ajoute les résultats successifs à res.
	// res doit être initialisé.
	static void mult(BigFloat& a, BigFloat& b, BigFloat& res) {
		for (int i = BIG_FLOAT_SIZE - 1; i >= 0; i--)
			multDigit(a, b, i, res);
		res.pos = (a.pos == b.pos);
	}

	// Multiplication d’un BigFloat par un float.
	// (utilisé pour les points de départ)
	// (peut être amélioré en évitant de repasser par un BigFloat)
	static void mult(float a, BigFloat& b, BigFloat& res) {
		BigFloat aBig(a);
		mult(aBig, b, res);
	}

};