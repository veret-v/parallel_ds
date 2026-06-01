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
#include <set>
#include <queue>
#include <algorithm>
#include <memory>

#include "linalg.hpp"
#include "valuesVector.hpp"
#include "matrix.hpp"

#include "../common/problem.hpp"
#include "../common/types.hpp"
#include "../common/LPsolution.hpp"
#include "../common/timeProfiler.hpp"
#include "../common/baseDualSimplex.hpp"


class SequentialDualSimplex : public BaseDualSimplex<Matrix, ValuesVector, IndexVector>
{
protected:
    std::vector<EtaMatrix> B_eta_repr;
    std::unique_ptr<TimeProfiler> _timer;

    
    PresolverMethods stringToPreSolverMethod(const std::string& method_name) override;

    bool callDualSolver() override;
    bool callPrimalSolver() override;

    void solveLinSys(
        ValuesVector&& rhs, 
        ValuesVector& sol,
        bool transpose
    ) override;

    void solveLinSys(
        ValuesVector& rhs, 
        ValuesVector& sol,
        bool transpose
    ) override;


    void BTran(int p_idx, ValuesVector& rho) override;
    void pivotRow(ValuesVector& rho, ValuesVector& alpha) override;
    void FTran(int q, ValuesVector& alpha_q) override;
    void initReducedCosts() override;
    void initPhase1PricingVector(ValuesVector& f, IndexVector& inf_u_indexes, IndexVector& inf_l_indexes, double& Z) override;
    void reFactorize() override;
    void simpleReducedCostsUpate(const ValuesVector& alpha, int q_idx, int q, double theta) override;
    void updateAndChangeBasis(
        ValuesVector& f, ValuesVector& rho, const ValuesVector& alpha_q, 
        int p_idx, int p, int q_idx, int q, double theta_P
    ) override;
    void phase1UpdateAndChangeBasis(
        ValuesVector& f, ValuesVector& rho, const ValuesVector& alpha_q, 
        int p_idx, int p, int q_idx, int q, double theta_P
    ) override;

    void dualSimplexInit() override;

public:
    using BaseDualSimplex::BaseDualSimplex;

    void initDualSimplex();
};


