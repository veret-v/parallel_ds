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


namespace linalg
{
    ValuesVector unit(const int size, const int p);

    void PFIsolve(
        const std::vector<EtaMatrix>& A, 
        const ValuesVector& b, 
        ValuesVector& sol, 
        const bool transpose
    );
} // namespace linalg

