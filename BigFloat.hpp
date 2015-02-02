#pragma once

#include "Parametres.hpp"
#include <stdint.h>
#include <string>
#include <iostream>

class BigFloat {

public:
	uint32_t* decimals; // Chiffres en base 32
	uint32_t base; // Chiffre avant la virgule

	BigFloat() {
		decimals = new uint32_t[BIG_FLOAT_SIZE];
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			decimals[i] = 0;
		base = 0;
	}
	~BigFloat() {
		delete[] decimals;
	}
	void display() {
		std::cout << base << ",";
		for (int i = 0; i < BIG_FLOAT_SIZE; i++)
			std::cout << decimals[i];
		std::cout << std::endl;
	}

	//void add(BigFloat& a, BigFloat& b, BigFloat& res) {
	//	for (int i = 0; i < BIG_FLOAT_SIZE; i++)
	//		for (int j = 0)
	//}
	//
	//void mult(BigFloat& a, BigFloat& b, BigFloat& res) {
	//	for (int i = 0; )
	//}

};