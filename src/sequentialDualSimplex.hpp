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

#include "problem.hpp"
#include "linalg.hpp"
#include "valuesVector.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include "LPsolution.hpp"
#include "baseDualSimplex.hpp"

#define EPS_BOUND 1e-10
#define EPS_ALPHA 1e-8
#define EPS_D     1e-7
#define EPS_COSTS 1e-8
#define EPS_A     1e-5
#define EPS_Z     1e-12
#define REFACT_ERR 1e-9

#define PERTURB_RATIO 0.25
#define PSI           1e-5
#define MAX_CYCLE     5
#define REFACT_FREQ   200


class SequentialDualSimplex : public BaseDualSimplex
{
protected:

    Phase1OutStatus callPresolver(const PresolverMethods method) override;
    bool callDualSolver(const SolverMethods method) override;
    bool callPrimalSolver() override;
    
    SolverMethods    stringToSolverMethod(const std::string& method_name) override;
    PresolverMethods stringToPreSolverMethod(const std::string& method_name) override;

    Phase1OutStatus minimizeDualInfeasibility();
    Phase1OutStatus panMathod();

    bool simpleRatioMethod();
    bool elaboratedMethod();

    ValuesVector initBetaWeights();

public:
    using BaseDualSimplex::BaseDualSimplex;

};


