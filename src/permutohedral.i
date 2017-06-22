%module permutohedral

%{
#define SWIG_FILE_WITH_INIT	
#include "permutohedral.h"
%}

%include "numpy.i"

%init %{
	import_array();	
%}

%apply (float* IN_ARRAY1, int DIM1) {(float *img, int nn)};
%apply (unsigned long long* ARGOUT_ARRAY1, int DIM1) {(size_t *output, int nn)};
%apply (float* ARGOUT_ARRAY1, int DIM1) {(float *output, int nn)};

%include "permutohedral.h"