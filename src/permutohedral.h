#pragma once
#ifndef PERMUTOHEDRAL_LATTICE_H
#define PERMUTOHEDRAL_LATTICE_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <map>

//#ifdef WIN32
//#include "win32time.h"
//#else
//#include <sys/time.h>
//#endif

using namespace std;

/***************************************************************/
/* Author: Yuan Liang @ Michigan State University              */
/* Email: liangy11@msu.edu                                     */
/* Most codes are from the codes of the original paper         */
/***************************************************************/

/***************************************************************/
/* Permutohegral Lattice implementation of Bilateral Filtering */
/* Generate the transformation matrix for splat and blur       */
/* The splat calculates the enclosing lattice points           */
/* and their corresponding weights for each pixel              */
/* Blur calculates the neighbors of each nonzero lattice points*/
/* along each basis in the hyperplane                          */
/* Slice is the transpose of splat                             */
/***************************************************************/

class PermutohedralLattice {

public:	
	/* Constructor
	*     d_ : dimensionality of key vectors
	* nData_ : number of points in the input
	*/
	PermutohedralLattice(int d_, int nData_);

	~PermutohedralLattice();
	
	/* generating the splat vector (matrix)
	 * the slicing matrix is identical to splat matrix
	 */
	void generate_splat_vector(float *img, int nn);

	// the return APIs
	void get_enclosing_simplices(size_t *output, int nn);

	void get_weights(float *output, int nn);

	size_t get_lattice_points();

	void get_blur(size_t* output, int nn);

	/* generate d+1 blurring matrices, one for each basis */
	void generate_blur_vector();

private:

	/* Performs splatting with given position and value vectors */
	void splat(float *position, size_t *enclosing_simplex, float *b_weights);

	int d, nData;
	float *elevated, *scaleFactor, *barycentric;
	short *canonical;
	short *key;

	// newly added
	// for splat
	map<string, size_t> simplex_dict; // the dictionary of simplex points and their indices
	size_t  *enclosing_simplex_vector;
	float *b_weights_vector;
	// for blur
	size_t *blur_vector;


public:
	char  *rank;
	short *greedy;
};

#endif
