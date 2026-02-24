#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_set>

#include "matrix.hpp"
#include "valuesVector.hpp"

#include "../common/types.hpp"

#define EPS_ZERO 1e-8
#define EPS_SWAP 1e-6


namespace linalg
{
    ValuesVector unit(const int size, const int p);

    ValuesVector PFIsolve(const std::vector<EtaMatrix>& A, const ValuesVector& b, const bool transpose);
    bool PFIdecompose(const Matrix& A, std::vector<EtaMatrix>& decomposed);
} // namespace linalg

