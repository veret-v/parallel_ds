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

#define EPS_BOUND 1e-10
#define EPS_ALPHA 1e-8
#define EPS_D     1e-7
#define EPS_COSTS 1e-8
#define EPS_A     1e-5
#define EPS_Z     1e-12
#define REFACT_ERR 1e-9

#define PERTURB_RATIO 0.25
#define PSI           1e-5
#define RESTART_CYCLE 1000

// #define DEBUG


class BaseDualSimplex
{
protected:
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

    PresolverMethods presolver_method;
    SolverMethods solver_method;

    ValuesVector x;
    ValuesVector d;
    ValuesVector original_costs;

    IndexVector basis_indexes;
    IndexVector non_basis_indexes;

    std::vector<EtaMatrix> B_eta_repr;
    Matrix B;
    Matrix AN;

    double obj_func_val;


    virtual SolverMethods    stringToSolverMethod(const std::string& method_name);
    virtual PresolverMethods stringToPreSolverMethod(const std::string& method_name);

    virtual Phase1OutStatus callPresolver(const PresolverMethods method);
    virtual bool callDualSolver(const SolverMethods method);
    virtual bool callPrimalSolver();

    void calcDualInfeasible();
    size_t counterDualInfeasible() const;

    void perturbCosts();
    void initDualSimplex();
    void randomBasis();

    bool checkPrimalFeasible() const;
    bool checkDualFeasible()   const;
    bool checkPerturbNeed()    const;

    double getWeight(const size_t i) const;
    size_t calcNonzeroInColumn(const size_t i) const;

    bool minLex(const ValuesVector& a, const ValuesVector& b);
    ValuesVector prepareForLex(const ValuesVector& a, const size_t idx);

    bool setDelta(const size_t& j, double& delta, bool& is_lower);
    bool setRatioTestCandidates(IndexVector& F,const ValuesVector& tmp_alpha_p);

public:
    BaseDualSimplex(Problem& _problem);
    
    Phase1OutStatus presolve(const std::string& presolver_method_name);
    LPsolution solve(const std::string& method_name);
};

