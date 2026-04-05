#include "valuesVector.hpp"
#include "matrix.hpp"

#include "../common/problem.hpp"
#include "../common/baseDualSimplex.hpp"


template class Problem<Matrix, ValuesVector>;
template class BaseDualSimplex<Matrix, ValuesVector, IndexVector>;