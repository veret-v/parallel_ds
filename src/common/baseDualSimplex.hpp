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
#include "types.hpp"
#include "LPsolution.hpp"

#include "../cpu/linalg.hpp"
#include "../cpu/valuesVector.hpp"
#include "../cpu/matrix.hpp"

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


template <typename MatrixType, typename VectorType, typename IndexVectorType>
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

    int maxcycle;
    int non_basis_size;
    int basis_size;

    Problem<MatrixType, VectorType>* problem;

    LPsolution solution;

    PresolverMethods presolver_method;
    SolverMethods solver_method;

    VectorType x;
    VectorType d;
    VectorType original_costs;

    IndexVectorType basis_indexes;
    IndexVectorType non_basis_indexes;

    MatrixType B;

    double obj_func_val;

    virtual SolverMethods    stringToSolverMethod(const std::string& method_name);
    virtual PresolverMethods stringToPreSolverMethod(const std::string& method_name);

    virtual Phase1OutStatus callPresolver(const PresolverMethods method);
    virtual bool callDualSolver(const SolverMethods method);
    virtual bool callPrimalSolver();

    virtual void initDualSimplex();

    void calcDualInfeasible();
    int counterDualInfeasible() const;

    void perturbCosts();
    void randomBasis();

    bool checkPrimalFeasible() const;
    bool checkDualFeasible()   const;
    bool checkPerturbNeed()    const;

    double getWeight(const int i) const;
    int calcNonzeroInColumn(const int i) const;

    bool setDelta(const int& j, double& delta, bool& is_lower);
    bool setRatioTestCandidates(IndexVector& F,const VectorType& tmp_alpha_p);

public:
    BaseDualSimplex(Problem<MatrixType, VectorType>& _problem);
    
    Phase1OutStatus presolve(const std::string& presolver_method_name);
    LPsolution solve(const std::string& method_name);
};

