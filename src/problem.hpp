#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream>
#include <string>
#include <cmath>
#include <unordered_map>

#include "linalg.hpp"
#include "valuesVector.hpp"
#include "matrix.hpp"
#include "types.hpp"


#define EPS_COND 1e6


class sequentialDualSimplex;
class LPparser;


class Problem
{

friend class sequentialDualSimplex;
friend class LPparser;

private:
   
    size_t problem_size;
    size_t constraints_size;
    size_t logicals_size;

    BoundaryTypeVector bound_type;

    ValuesVector costs;

    ValuesVector lower_bound;
    ValuesVector upper_bound;

    ValuesVector RHS;
    
    Matrix A;


    void checkConstraints();
    void transformToComputeForm(
        const ValuesVector &lower_range,
        const ValuesVector &upper_range,
        const BoundaryTypeVector &range_type
    );

    BoundaryType stringToBoundaryType(const std::string& bound_type_name) const;
    std::string boundaryTypeToString(const BoundaryType& bound_type_name) const;
    void setBoundary(const size_t& idx, const std::string& bound_type_name);

    // void scaleProblem();

public:
    Problem(const Problem &problem);
    Problem() {};
    Problem(
        const BoundaryTypeVector &bound_type,
        const BoundaryTypeVector &range_type,
        const ValuesVector &costs,
        const ValuesVector &lower_range,
        const ValuesVector &upper_range,
        const ValuesVector &lower_bound,
        const ValuesVector &upper_bound,
        const Matrix &_A
    );

    void show();
};
