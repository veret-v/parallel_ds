#include "cudaSparseMatrix.hpp"
#include "cudaDenseVector.hpp"
#include "cudaDataDenseVector.hpp"
#include "cudaIndexVector.hpp"

#include "../common/problem.hpp"
#include "../common/baseDualSimplex.hpp"


template class Problem<CudaSparseMatrix, CudaDataDenseVector>;
template class BaseDualSimplex<CudaSparseMatrix, CudaDataDenseVector, CudaIndexVector>;
