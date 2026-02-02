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
    ValuesVector unit(const size_t size, const size_t p);

    ValuesVector PFIsolve(const std::vector<EtaMatrix>& A, const ValuesVector& b, const bool transpose);
    bool PFIdecompose(const Matrix& A, std::vector<EtaMatrix>& decomposed);
    ValuesVector LUsolve(const std::vector<EtaMatrix>& A, const ValuesVector& b, const bool transpose);
    bool LUdecompose(const Matrix& A, std::vector<EtaMatrix>& decomposed);
} // namespace linalg

