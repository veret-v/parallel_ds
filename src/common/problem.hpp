#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream>
#include <string>
#include <cmath>
#include <unordered_map>

#include "types.hpp"

#include "../cpu/linalg.hpp"
#include "../cpu/valuesVector.hpp"
#include "../cpu/matrix.hpp"


#define EPS_COND 1e6


class SequentialDualSimplex;
class ParallelDualSimplex;
class BaseDualSimplex;
class CudaDualSimplex;


template <typename MatrixType, typename VectorType>
class Problem
{

friend class SequentialDualSimplex;
friend class ParallelDualSimplex;
friend class BaseDualSimplex;
friend class CudaDualSimplex;

private:
   
    int problem_size;
    int constraints_size;
    int logicals_size;

    BoundaryTypeVector bound_type;

    VectorType costs;

    VectorType lower_bound;
    VectorType upper_bound;

    VectorType RHS;
    
    MatrixType A;

    void checkConstraints();
    void transformToComputeForm(
        const VectorType &lower_range,
        const VectorType &upper_range,
        const BoundaryTypeVector &range_type
    );

    BoundaryType stringToBoundaryType(const std::string& bound_type_name) const;
    std::string boundaryTypeToString(const BoundaryType& bound_type_name) const;
    void setBoundary(const int& idx, const std::string& bound_type_name);

    BoundaryType mpsTypeToBoundaryType(const char bound_type_name) const;
    inline bool isinf_bound(const double x) const {return (x > bound_inf || x < -bound_inf) ? true : false;};

public:
    Problem(const Problem &problem);
    Problem(const std::string& file_name);
    Problem() {};
    Problem(
        const BoundaryTypeVector &bound_type,
        const BoundaryTypeVector &range_type,
        const VectorType &costs,
        const VectorType &lower_range,
        const VectorType &upper_range,
        const VectorType &lower_bound,
        const VectorType &upper_bound,
        const MatrixType &_A
    );

    void show();
};
