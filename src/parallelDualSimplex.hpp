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

#include "problem.hpp"
#include "linalg.hpp"
#include "valuesVector.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include "LPsolution.hpp"

#define EPS_BOUND 1e-10
#define EPS_ALPHA 1e-8
#define EPS_D     1e-7
#define EPS_COSTS 1e-8
#define EPS_A     1e-5
#define EPS_Z     1e-12
#define REFACT_ERR 1e-9

#define PERTURB_RATIO 0.25
#define PSI           1e-5
#define CAND_RATIO    0.95
#define MAX_CYCLE     5
#define REFACT_FREQ   50

#define DEBUG

class parallelDualSimplex
{
private:
    enum class SolverMethods
    {
        elaboratedMethod,
        UNKNOWN
    };

    enum class PresolverMethods
    {
        minDualInfeasibility,
        UNKNOWN
    };

    enum class Phase1OutStatus
    {
        Solved,
        DualInfeas,
        NeedRestart
    };

    bool perturbed = false;

    size_t maxcycle;

    size_t non_basis_size;

    Problem* problem;

    LPsolution solution;

    ValuesVector x;
    ValuesVector d;
    ValuesVector original_costs;

    IndexVector basis_indexes;
    IndexVector non_basis_indexes;

    std::vector<EtaMatrix> B_eta_repr;
    Matrix B;
    Matrix AN;

    double obj_func_val;


    SolverMethods    stringToSolverMethod(const std::string& method_name);
    PresolverMethods stringToPreSolverMethod(const std::string& method_name);

    Phase1OutStatus presolve(const std::string& presolver_method_name);
    Phase1OutStatus minimizeDualInfeasibility();

    bool elaboratedMethod();

    Phase1OutStatus callPresolver(const PresolverMethods method);
    bool callDualSolver(const SolverMethods method);
    bool callPrimalSolver();

    void calcDualInfeasible();
    size_t counterDualInfeasible() const;

    void perturbCosts();

    bool checkPrimalFeasible() const;
    bool checkDualFeasible()   const;
    bool checkPerturbNeed()    const;

    ValuesVector initBetaWeights();

    double getWeight(const size_t i) const;
    size_t calcNonzeroInColumn(const size_t i) const;

    bool minLex(const ValuesVector& a, const ValuesVector& b);
    ValuesVector prepareForLex(const ValuesVector& a, const size_t idx);

    bool setDelta(const size_t& j, double& delta, bool& is_lower);
    bool setRatioTestCandidates(IndexVector& F,const ValuesVector& tmp_alpha_p);

public:
    parallelDualSimplex(
        Problem& _problem,
        const std::string& presolver_method_name
    );

    LPsolution solve(const std::string& method_name);
};

