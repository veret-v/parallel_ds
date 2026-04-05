#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream>
#include <string>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <CoinMpsIO.hpp>

#include "types.hpp"
#include "LPsolution.hpp"

#include "../cpu/linalg.hpp"
#include "../cpu/valuesVector.hpp"
#include "../cpu/matrix.hpp"


#ifdef WITH_CUDA
    #include "../gpu/cudaSparseMatrix.hpp"
    #include "../gpu/cudaDenseVector.hpp"
    #include "../gpu/cudaDataDenseVector.hpp"
    #include "../gpu/cudaIndexVector.hpp"
#endif

#define EPS_COND  1e6
#define BOUND_INF 1e+100


template <typename MatrixType, typename VectorType>
class Problem
{
public:
    bool scaled = false;

    int problem_size;
    int constraints_size;
    int logicals_size;

    LPsolution solution;

    VectorType scale_rows;
    VectorType scale_cols;

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
        const BoundaryTypeVector &range_type,
        Matrix& A_buff
    );
    void reduce(
        VectorType &lower_range,
        VectorType &upper_range,
        BoundaryTypeVector &range_type,
        Matrix& A_buff
    );

    void scale(
        Matrix& A_buff,
        VectorType &lower_range,
        VectorType &upper_range
    );
    bool checkWellScaled(Matrix& A_buff);

    BoundaryType stringToBoundaryType(const std::string& bound_type_name) const;
    std::string boundaryTypeToString(const BoundaryType& bound_type_name) const;
    void setBoundary(const int& idx, const std::string& bound_type_name);

    BoundaryType mpsTypeToBoundaryType(const char bound_type_name) const;
    inline bool isinf_bound(const double x) const {return (x > BOUND_INF || x < -BOUND_INF) ? true : false;};

    Problem(const Problem &problem);
    Problem() {};
    Problem(
        const BoundaryTypeVector &bound_type,
        const BoundaryTypeVector &range_type,
        const VectorType &costs,
        const VectorType &lower_range,
        const VectorType &upper_range,
        const VectorType &lower_bound,
        const VectorType &upper_bound,
        const Matrix &_A
    );

    void readMps(const std::string& file_name);
    void show();

    LPsolution getSolution();
};

#include "problem.tpp" 