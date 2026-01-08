#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream>
#include <vector>
#include <cmath>

#include "matrix.hpp"
#include "valuesVector.hpp"
#include "types.hpp"


#define EPS_ZERO 1e-8
#define EPS_SWAP 1e-6


namespace linalg
{
    Matrix ones(const size_t n);
    ValuesVector unit(const size_t size, const size_t p);
    ValuesVector PFIsolve(const std::vector<EtaMatrix>& A, const ValuesVector& b, const bool transpose);
    bool PFIdecompose(const Matrix& A, std::vector<EtaMatrix>& decomposed);
    bool checkPFIdecompose(const std::vector<EtaMatrix>& B_eta_repr, const Matrix& B);
    void dotEtaMatrix(const EtaMatrix& eta_matrix, Matrix& matrix);
} // namespace linalg

