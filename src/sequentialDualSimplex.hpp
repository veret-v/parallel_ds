#pragma once

#include <string>
#include <unordered_map>
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

#define EPS_BOUND 1e-10
#define EPS_ALPHA 1e-8
#define EPS_D     1e-8
#define EPS_COSTS 1e-8
#define EPS_A     1e-8

#define PERTURB_RATIO 0.25
#define PSI           1e-5
#define MAX_CYCLE     5


class LPsolution
{
public:
    double Z;
    bool solved;
    ValuesVector x;
    std::string message;

    LPsolution(
        const double _Z, 
        const ValuesVector& _x, 
        const bool _solved, 
        const std::string& _message
    ) : Z(_Z), x(_x), solved(_solved), message(_message) {};
    LPsolution(
        const LPsolution& solution
    ) : LPsolution(solution.Z, solution.x, solution.solved, solution.message) {};
    LPsolution() : LPsolution(0, ValuesVector(), false, "") {};

    void show();
};


class sequentialDualSimplex
{
private:
    enum class SolverMethods
    {
        simpleRatio,
        elaboratedMethod,
        UNKNOWN
    };

    enum class PresolverMethods
    {
        minDualInfeasibility,
        panMethod,
        UNKNOWN
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

    void presolve(const std::string& presolver_method_name);

    bool minimizeDualInfeasibility();
    bool panMathod();

    bool simpleRatioMethod();
    bool elaboratedMethod();

    bool callPresolver(const PresolverMethods method);
    bool callDualSolver(const SolverMethods method);
    bool callPrimalSolver();

    void calcDualInfeasible();

    void perturbCosts();

    bool checkPrimalFeasible() const;
    bool checkDualFeasible()   const;
    bool checkPerturbNeed()    const;

    double getWeight(const size_t i) const;
    size_t calcNonzeroInColumn(const size_t i) const;

    bool minLex(const ValuesVector& a, const ValuesVector& b);
    ValuesVector prepareForLex(const ValuesVector& a, const size_t idx);

public:
    sequentialDualSimplex(
        Problem& _problem,
        const std::string& presolver_method_name
    );

    LPsolution solve(const std::string& method_name);
};

