#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <chrono>
#include <set>
#include <queue>
#include <algorithm>

#include <omp.h>

#include "linalg.hpp"
#include "valuesVector.hpp"
#include "matrix.hpp"

#include "../common/problem.hpp"
#include "../common/types.hpp"
#include "../common/LPsolution.hpp"
#include "../common/baseDualSimplex.hpp"


class ParallelDualSimplex : public BaseDualSimplex<Matrix, ValuesVector, IndexVector>
{
protected:
    std::vector<EtaMatrix> B_eta_repr;
   
    SolverMethods    stringToSolverMethod(const std::string& method_name) override;
    PresolverMethods stringToPreSolverMethod(const std::string& method_name) override;

    Phase1OutStatus minimizeDualInfeasibility();

    bool elaboratedMethod();

    Phase1OutStatus callPresolver(const PresolverMethods method) override;
    bool callDualSolver(const SolverMethods method) override;
    bool callPrimalSolver() override;
    void initReducedCosts(ValuesVector& vec) override;

    ValuesVector initBetaWeights();

     void solveLinSys(
        ValuesVector&& rhs, 
        ValuesVector& sol,
        bool transpose
    );

    void solveLinSys(
        ValuesVector& rhs, 
        ValuesVector& sol,
        bool transpose
    );

public:
    using BaseDualSimplex::BaseDualSimplex;

    void initDualSimplex();

};

