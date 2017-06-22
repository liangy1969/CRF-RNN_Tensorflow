#include "permutohedral.h"
#include <string.h>

PermutohedralLattice::PermutohedralLattice(int d_, int nData_) :
	d(d_), nData(nData_), enclosing_simplex_vector(0), b_weights_vector(0),  blur_vector(0){

	// Allocate storage for various arrays
	elevated = new float[d + 1];
	scaleFactor = new float[d];

	greedy = new short[d + 1];
	rank = new char[d + 1];
	barycentric = new float[d + 2];

	canonical = new short[(d + 1)*(d + 1)];
	key = new short[d + 1];

	// compute the coordinates of the canonical simplex, in which
	// the difference between a contained point and the zero
	// remainder vertex is always in ascending order. (See pg.4 of paper.)
	for (int i = 0; i <= d; i++) {
		for (int j = 0; j <= d - i; j++)
			canonical[i*(d + 1) + j] = i;
		for (int j = d - i + 1; j <= d; j++)
			canonical[i*(d + 1) + j] = i - (d + 1);
	}

	// Compute parts of the rotation matrix E. (See pg.4-5 of paper.)      
	for (int i = 0; i < d; i++) {
		// the diagonal entries for normalization
		scaleFactor[i] = 1.0f / (sqrtf((float)(i + 1)*(i + 2)));

		/* We presume that the user would like to do a Gaussian blur of standard deviation
		* 1 in each dimension (or a total variance of d, summed over dimensions.)
		* Because the total variance of the blur performed by this algorithm is not d,
		* we must scale the space to offset this.
		*
		* The total variance of the algorithm is (See pg.6 and 10 of paper):
		*  [variance of splatting] + [variance of blurring] + [variance of splatting]
		*   = d(d+1)(d+1)/12 + d(d+1)(d+1)/2 + d(d+1)(d+1)/12
		*   = 2d(d+1)(d+1)/3.
		*
		* So we need to scale the space by (d+1)sqrt(2/3).
		*/
		scaleFactor[i] *= (d + 1)*sqrtf(2.0 / 3);
	}
}

PermutohedralLattice::~PermutohedralLattice() {
	// clear the memory
	delete elevated;
	delete scaleFactor;
	delete greedy;
	delete rank;
	delete barycentric;
	delete canonical;
	delete key;

	if (enclosing_simplex_vector) delete enclosing_simplex_vector;
	if (b_weights_vector) delete b_weights_vector;
	if (blur_vector) delete blur_vector;
}

void PermutohedralLattice::generate_splat_vector(float *img, int nn) {

	if (nn != d*nData) {
		printf("Invalid input size");
		return;
	}
	
	// clear the memory
	simplex_dict.clear();

	if (enclosing_simplex_vector) {
		
		delete enclosing_simplex_vector;
	}

	enclosing_simplex_vector = new size_t[nData*(d + 1)];

	
	if (b_weights_vector) {
		delete b_weights_vector;
	}
	
	b_weights_vector = new float[nData*(d + 1)];

	

	for (size_t i = 0; i < nData; ++i) {
		float *position = img + i*d;
		size_t *enclosing_simplex = enclosing_simplex_vector + i*(d + 1);
		float *b_weights = b_weights_vector + i*(d + 1);
		splat(position, enclosing_simplex, b_weights);
	}
}

// the return APIs
void PermutohedralLattice::get_enclosing_simplices(size_t *output, int nn) {

	if (nn != nData*(d + 1)) {
		printf("Invalid output size");
		return;
	}
	//
	for (int i = 0; i < nData*(d + 1); i++) output[i] = enclosing_simplex_vector[i];
}

void PermutohedralLattice::get_weights(float *output, int nn) {

	if (nn != nData*(d + 1)) {
		printf("Invalid output size");
		return;
	}
	//
	for (int i = 0; i < nData*(d + 1); i++) output[i] = b_weights_vector[i];
}

size_t PermutohedralLattice::get_lattice_points() {
	//
	return simplex_dict.size();
}

void PermutohedralLattice::get_blur(size_t* output, int nn) {

	//
	size_t n_lattice = simplex_dict.size();

	if (nn != (d + 1) * 3 * n_lattice) {
		printf("Invalid output size");
		return;
	}

	for (int i = 0; i < (d + 1) * 3 * n_lattice; i++) output[i] = blur_vector[i];
}

void PermutohedralLattice::generate_blur_vector() {

	// the number of lattice points
	size_t n_lattice = simplex_dict.size();

	// clear the memory
	if (blur_vector) delete blur_vector;
	blur_vector = new size_t[(d + 1) * 3 * n_lattice];

	// Prepare arrays
	short *neighbor1 = new short[d + 1];
	short *neighbor2 = new short[d + 1];

	size_t *blur_vector_ptr = 0;

	// For each of d+1 axes,
	for (int j = 0; j <= d; j++) {

		blur_vector_ptr = blur_vector + j * 3 * n_lattice;

		map<string, size_t>::iterator itr_end = simplex_dict.end();

		// For each vertex in the lattice,
		for (map<string, size_t>::iterator itr = simplex_dict.begin(); itr != itr_end; ++itr) {
			// blur in dimension j
			size_t ind_cur = itr->second; // the index of current vertex

			short *key = (short *)(itr->first.c_str()); // keys to current vertex

			for (int k = 0; k < d; k++) {
				neighbor1[k] = key[k] + 1;
				neighbor2[k] = key[k] - 1;
			}

			neighbor1[j] = key[j] - d;
			neighbor2[j] = key[j] + d; // keys to the neighbors along the given axis.

			string neighbor1_loc_string = string((char *)neighbor1, (size_t)(d * sizeof(short) / sizeof(char)));
			string neighbor2_loc_string = string((char *)neighbor2, (size_t)(d * sizeof(short) / sizeof(char)));

			// check whether the two neighbors exists
			size_t neighbor1_ind = n_lattice;
			map<string, size_t>::iterator itr_neighbor1 = simplex_dict.find(neighbor1_loc_string);
			if (itr_neighbor1 != itr_end) neighbor1_ind = itr_neighbor1->second;

			size_t neighbor2_ind = n_lattice;
			map<string, size_t>::iterator itr_neighbor2 = simplex_dict.find(neighbor2_loc_string);
			if (itr_neighbor2 != itr_end) neighbor2_ind = itr_neighbor2->second;

			// store the results
			blur_vector_ptr[0] = ind_cur;
			blur_vector_ptr[1] = neighbor1_ind;
			blur_vector_ptr[2] = neighbor2_ind;

			blur_vector_ptr += 3;
		}

	}
	delete neighbor1;
	delete neighbor2;
}

void PermutohedralLattice::splat(float *position, size_t *enclosing_simplex, float *b_weights) {

	// first rotate position into the (d+1)-dimensional hyperplane
	elevated[d] = -d*position[d - 1] * scaleFactor[d - 1];
	for (int i = d - 1; i > 0; i--)
		elevated[i] = (elevated[i + 1] -
			i*position[i - 1] * scaleFactor[i - 1] +
			(i + 2)*position[i] * scaleFactor[i]);
	elevated[0] = elevated[1] + 2 * position[0] * scaleFactor[0];

	// prepare to find the closest lattice points
	float scale = 1.0f / (d + 1);
	char * myrank = rank;
	short * mygreedy = greedy;

	// greedily search for the closest zero-colored lattice point
	int sum = 0;
	for (int i = 0; i <= d; i++) {
		float v = elevated[i] * scale;
		float up = ceilf(v)*(d + 1);
		float down = floorf(v)*(d + 1);

		if (up - elevated[i] < elevated[i] - down) mygreedy[i] = (short)up;
		else mygreedy[i] = (short)down;

		sum += mygreedy[i];
	}
	sum /= d + 1;

	// rank differential to find the permutation between this simplex and the canonical one.
	// (See pg. 3-4 in paper.)
	memset(myrank, 0, sizeof(char)*(d + 1));
	for (int i = 0; i < d; i++)
		for (int j = i + 1; j <= d; j++)
			if (elevated[i] - mygreedy[i] < elevated[j] - mygreedy[j]) myrank[i]++; else myrank[j]++;

	if (sum > 0) {
		// sum too large - the point is off the hyperplane.
		// need to bring down the ones with the smallest differential
		for (int i = 0; i <= d; i++) {
			if (myrank[i] >= d + 1 - sum) {
				mygreedy[i] -= d + 1;
				myrank[i] += sum - (d + 1);
			}
			else
				myrank[i] += sum;
		}
	}
	else if (sum < 0) {
		// sum too small - the point is off the hyperplane
		// need to bring up the ones with largest differential
		for (int i = 0; i <= d; i++) {
			if (myrank[i] < -sum) {
				mygreedy[i] += d + 1;
				myrank[i] += (d + 1) + sum;
			}
			else
				myrank[i] += sum;
		}
	}

	// Compute barycentric coordinates (See pg.10 of paper.)
	memset(barycentric, 0, sizeof(float)*(d + 2));
	for (int i = 0; i <= d; i++) {
		barycentric[d - myrank[i]] += (elevated[i] - mygreedy[i]) * scale;
		barycentric[d + 1 - myrank[i]] -= (elevated[i] - mygreedy[i]) * scale;
	}
	barycentric[0] += 1.0f + barycentric[d + 1];

	size_t dict_size = simplex_dict.size();
	// Splat the value into each vertex of the simplex, with barycentric weights.
	for (int remainder = 0; remainder <= d; remainder++) {
		// for the remainder-th enclosing simplex point
		// Compute the location of the lattice point explicitly (all but the last coordinate - it's redundant because they sum to zero)
		for (int i = 0; i < d; i++)
			key[i] = mygreedy[i] + canonical[remainder*(d + 1) + myrank[i]];

		// convert the location of simplex point into the corresponding string expression
		string loc_string = string((char *)key, (size_t)(d * sizeof(short) / sizeof(char)));

		// check whether the point exists in the map
		map<string, size_t>::iterator it = simplex_dict.find(loc_string);

		if (it != simplex_dict.end()) {
			//if the point does exist
			enclosing_simplex[remainder] = it->second;
		}
		else {
			// add the new point in
			simplex_dict[loc_string] = dict_size;
			enclosing_simplex[remainder] = dict_size;
			dict_size = dict_size + 1;
			// printf("%d",dict_size);
		}

		// store barycentric weights.
		b_weights[remainder] = barycentric[remainder];

	}
}