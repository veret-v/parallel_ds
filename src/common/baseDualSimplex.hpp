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
#define EPS_D     1e-8
#define EPS_COSTS 1e-8
#define EPS_A     1e-5
#define EPS_Z     1e-10
#define REFACT_ERR 1e-9
#define EPS_R      1e-9
#define EPS_P      1e-7
#define EPS_F      1e-10

#define PERTURB_RATIO 0.25
#define PSI           1e-5
#define RESTART_CYCLE 1000
#define CUT_OFF       0.95

#define MAX_CYCLE      5 
#define NEED_RESTART   40
#define REFACT_FREQ    150
#define CHECK_DUAL_INF 500
#define RESTART_SIZE   10
#define MAX_ITER       40000
#define INF            1e+25

#define DEBUG


template <typename MatrixType, typename VectorType, typename IndexVectorType>
class BaseDualSimplex
{
protected:
    //--------------------------------------------------------------------------------
    // Params
    //--------------------------------------------------------------------------------
    using PricingInfo       = std::tuple<int, int, double, bool>;
    using Phase1PricingInfo = std::tuple<int, int, bool>;
    using RatioTestInfo     = std::tuple<int, int>;

    bool perturbed = false;

    int maxcycle;
    int non_basis_size;
    int basis_size;
    int full_size;
    int iteration = 0;

    Problem<MatrixType, VectorType>* problem;

    PresolverMethods presolver_method;
    SolverMethods solver_method;

    VectorType x;
    VectorType d;
    VectorType original_costs;
    VectorType beta;

    IndexVectorType basis_indexes;
    IndexVectorType non_basis_indexes;

    MatrixType B;

    double obj_func_val;


    //--------------------------------------------------------------------------------
    // Protected Functions
    //--------------------------------------------------------------------------------

    // Phase 1 functions
    Phase1OutStatus minimizeDualInfeasibility();
    Phase1PricingInfo phase1Pricing(VectorType& f, std::unordered_set<int>& blocked_p);

    void minimizeDualInfeasibilityInit(VectorType& f);

    double infisibilityCorr(int q);

    virtual void initPhase1PricingVector(
        VectorType& f, IndexVector& inf_u_indexes, 
        IndexVector& inf_l_indexes, double& Z
    );
    virtual void phase1UpdateAndChangeBasis(
        VectorType& f, VectorType& rho, const VectorType& alpha_q, 
        int p_idx, int p, int q_idx, int q, double theta_P
    );


    // Algorithm steps
    virtual void BTran(int p_idx, VectorType& rho);
    virtual void pivotRow(VectorType& rho, VectorType& alpha);
    virtual void FTran(int q, VectorType& alpha_q);
    virtual void initReducedCosts();
    virtual void dualSimplexInit();
    virtual void reFactorize();
    virtual void simpleReducedCostsUpate(const VectorType& alpha, int p, int q, double theta);
    virtual void updateAndChangeBasis(
        VectorType& f, VectorType& rho, const VectorType& alpha_q, 
        int p_idx, int p, int q_idx, int q, double theta_P
    );


    // Convert string to exiting methods for presolver
    virtual PresolverMethods stringToPreSolverMethod(const std::string& method_name);


    // Phase 2 simple method for finding solution, based on article:
    // Dipl. Inform. Achim Koberstein. The Dual Simplex Method, Techniques for a fast and 
    // stable implementation: for a fast and stable implementation, November 2005
    virtual bool             callDualSolver();
    // Call priaml solver 
    virtual bool             callPrimalSolver();

    // Post processing
    void postProcess(bool status_code);


    // Linear system solution
    /*
    Solve system of linear equations by LU factorization and PFI updates.
    PBQ = LU -> B = P^-1LUQ^-1, 
    after column swap B0 -> B1 = E1 * B0 = E1 * P^-1LUQ^-1 - this is LU + PFI
    */
    virtual void solveLinSys(
        VectorType&& rhs, 
        VectorType& sol,
        bool transpose
    );
    /*
    Solve system of linear equations by LU factorization and PFI updates.
    PBQ = LU -> B = P^-1LUQ^-1, 
    after column swap B0 -> B1 = E1 * B0 = E1 * P^-1LUQ^-1 - this is LU + PFI
    */
    virtual void solveLinSys(
        VectorType& rhs, 
        VectorType& sol,
        bool transpos
    );

    // Pricing rules
    std::pair<bool, double> checkVarIsInfeas(int i);
    PricingInfo DSEPricing(std::unordered_set<int>& blocked_p);
    PricingInfo simplePricing(std::unordered_set<int>& blocked_p);

    //Ratio tests
    IndexVector phase1SetRatioTestCandidates(const VectorType& alpha);
    RatioTestInfo simpleRatioTest(IndexVector& F, const VectorType& alpha);


    // Choose presolver 
    Phase1OutStatus  callPresolver(const PresolverMethods method);


    // Set x non basis vars to extreme positions. Example: x > 0 extreme pos - x = 0;
    void setPrimalVars();


    //Perturb costs according to 
    // Dipl. Inform. Achim Koberstein. The Dual Simplex Method, Techniques for a fast and 
    // stable implementation: for a fast and stable implementation, November 2005
    void perturbCosts();

    // Check if costs need perturbation to prevent cycling
    bool checkPerturbNeed()    const;


    // Set random basis
    void randomBasis();


    // Initialize weights for DSE according to
    // John J. Forrest and Donald Goldfarb. Steepest-edge simplex algorithms for
    // linear programming. Math. Program., 57(3):341–374, 1992.
    void initBetaWeights(bool Ibasis);

    // Weight for perturbation depends on two goals "keep nonzero count low" and 
    // "resolver degeneracy"
    double getWeight(const int i)           const;


    // Calc dual infeasibility 
    void calcDualInfeasible();

    // COunt dual infeasibility 
    int counterDualInfeasible() const;
    int counterPrimalInfeasible() const;

    // Check primal feasibility in solver
    bool checkPrimalFeasible() const;

    // Check dual feasibility in solver
    bool checkDualFeasible()   const;

public:
    //--------------------------------------------------------------------------------
    // Public Functions
    //--------------------------------------------------------------------------------

    // Constructor for solver, additionally finds dual feasible basis
    BaseDualSimplex(Problem<MatrixType, VectorType>& _problem);
    
    // Dual simplex method: Phase 1(Find dual feasible basis)
    Phase1OutStatus presolve(const std::string& presolver_method_name);

    // Dual simplex method: Phase 2(Find solution)
    void solve();
};


#include "baseDualSimplex.tpp"