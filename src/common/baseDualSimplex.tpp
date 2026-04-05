#include "baseDualSimplex.hpp"


//----------------------------------------------------------------------------------------
// Convert string to exiting methods for presolver
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
PresolverMethods BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::stringToPreSolverMethod(
    const std::string& method_name
)
{
    return PresolverMethods::UNKNOWN;
}


//----------------------------------------------------------------------------------------
// Convert string to exiting methods for solver
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
SolverMethods BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::stringToSolverMethod(
    const std::string& method_name
)
{
    return SolverMethods::UNKNOWN;
}


//----------------------------------------------------------------------------------------
// Dual simplex method: Phase 1(Find dual feasible basis)
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
Phase1OutStatus BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::presolve(const std::string& presolver_method_name)
{
    std::cout << "Phase 1 : started" << std::endl;
    auto method = stringToPreSolverMethod(presolver_method_name);
    presolver_method = method;

    if (checkPerturbNeed())
        perturbCosts();

    Phase1OutStatus status = callPresolver(method);
    while(status == Phase1OutStatus::NeedRestart)
    {
        std::cout << "-- Restart" << std::endl;
        for (int i = 0; i < non_basis_size; i++)
            non_basis_indexes[i] = i;
    
        for (int i = 0; i < problem->constraints_size; i++)
            basis_indexes[i] = i + non_basis_size;

        status = callPresolver(method);
    }
    
    if (status == Phase1OutStatus::DualInfeas)
    {
        std::cout << "-- Problem is dual infeasible" << std::endl;
        return status;
    }
    std::cout << "-- Problem is feasible" << std::endl;


    for (auto i : non_basis_indexes)
    {
        switch (problem->bound_type[i])
        {
        case BoundaryType::Fixed:
            x[i] = problem->lower_bound[i];
            break;
            
        case BoundaryType::Free:
            x[i] = 0;
            break;

        case BoundaryType::Boxed:
            x[i] = (d[i] > 0) ? problem->lower_bound[i] : problem->upper_bound[i];
            break;

        case BoundaryType::Upper:
            x[i] = problem->upper_bound[i];
            break;

        case BoundaryType::Lower:
            x[i] = problem->lower_bound[i];
            break;
        }
    }
    std::cout << "-- Nonbasis variables bounds setted done" << std::endl;
    std::cout << "Phase 1 : ended" << std::endl;
    return status;
}


//----------------------------------------------------------------------------------------
// Choose presolver 
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
Phase1OutStatus BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::callPresolver(const PresolverMethods method)
{
    Phase1OutStatus status;
    switch (method) 
    {
        case PresolverMethods::UNKNOWN:
            throw "Unknown phase 1 solver method";
            break;
    }
    return status;
}


//----------------------------------------------------------------------------------------
// Choose solver 
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
bool BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::callDualSolver(const SolverMethods method)
{
    bool status;
    switch (method) 
    {
        case SolverMethods::UNKNOWN:
            throw "Unknown solver method";
            break;
    }
    return status;
}


//----------------------------------------------------------------------------------------
// Init reduced costs(d)
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::initReducedCosts(VectorType& vec)
{
}


//----------------------------------------------------------------------------------------
// Choose priaml solver 
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
bool BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::callPrimalSolver()
{
   return true;
}

//----------------------------------------------------------------------------------------
// Dual simplex method: Phase 2(Find solution)
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::solve(const std::string& method_name)
{
    std::cout << "Phase 2 : started" << std::endl;

    auto method = stringToSolverMethod(method_name);

    auto start = std::chrono::high_resolution_clock::now();

    bool status_code = callDualSolver(method);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
   
    std::cout << "time = " << duration_ms << std::endl;
   
    
    if (status_code && perturbed) // optimal solution case
    {
        std::cout << "-- Optimal solution obtained with perturbation" << std::endl;

        problem->costs = original_costs;
        VectorType buff(basis_size);
        initReducedCosts(buff);
       
        if (!checkDualFeasible()) 
        {
            callPrimalSolver();

            std::cout << "-- Correction in primal feasible and dual infeasible case done" << std::endl;
        }
        std::cout << "-- Solution obtained" << std::endl;

        problem->solution.x = x;
        problem->solution.Z += obj_func_val;

        std::cout << "Phase 2 : ended" << std::endl;

    }
    else if (!status_code && perturbed) // dual unbound case
    {
        std::cout << "-- Unbounded solution obtained with perturbation" << std::endl;

        // problem->costs = original_costs;
       
        // Phase1OutStatus status_code = callPresolver(presolver_method);
        // while (status_code == Phase1OutStatus::NeedRestart)
        // {    
        //     std::cout << "-- Restart phase 1" << std::endl;
        //     status_code = callPresolver(presolver_method);
        // }

        // if (status_code == Phase1OutStatus::Solved)
        //     callDualSolver(method);

        std::cout << "-- Correction done" << std::endl;
        std::cout << "-- Solution obtained" << std::endl;

        problem->solution.x = x;
        problem->solution.Z += obj_func_val;

        std::cout << "Phase 2 : ended" << std::endl;

    }
    else  // without perturbation case
    {
        std::cout << "-- Solution obtained" << std::endl;

        problem->solution.x = x;
        problem->solution.Z += obj_func_val;

        std::cout << "Phase 2 : ended" << std::endl;
    }
}


//----------------------------------------------------------------------------------------
// Constructor for solver, additionally finds dual feasible basis
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::BaseDualSimplex(Problem<MatrixType, VectorType>& _problem)
{
    std::cout << "Solver initialization : started" << std::endl;

    problem = &_problem;

    non_basis_size  = problem->problem_size - problem->constraints_size;
    basis_size      = problem->constraints_size;
    full_size       = problem->problem_size;
    
    basis_indexes       = IndexVector(problem->constraints_size);
    non_basis_indexes   = IndexVector(non_basis_size);

    for (int i = 0; i < non_basis_size; i++)
        non_basis_indexes[i] = i;
    
    for (int i = 0; i < problem->constraints_size; i++)
        basis_indexes[i] = i + non_basis_size;
}


//----------------------------------------------------------------------------------------
// Set random basis
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::randomBasis()
{
    IndexVector all_indexes(problem->problem_size);
    std::iota(all_indexes.begin(), all_indexes.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(all_indexes.begin(), all_indexes.end(), gen);

    basis_indexes = IndexVector(all_indexes.begin(), all_indexes.begin() + problem->constraints_size);
    non_basis_indexes = IndexVector(all_indexes.begin() + problem->constraints_size, all_indexes.end());
}


//----------------------------------------------------------------------------------------
// Check if costs need perturbation to prevent cycling
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
bool BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::checkPerturbNeed() const
{
    std::vector<double> unique_data;
    for (double c_i : problem->costs)
    {
        bool new_c = true;
        for (double uniq : unique_data)
        {
            if (fabs(c_i - uniq) < EPS_COSTS)
            {
                new_c = false;
                break;
            }            
        }
        if (new_c)
           unique_data.push_back(c_i);
    }
    
    return unique_data.size() < PERTURB_RATIO * problem->costs.getSize();

}


//----------------------------------------------------------------------------------------
//Perturb costs according to 
// Dipl. Inform. Achim Koberstein. The Dual Simplex Method, Techniques for a fast and 
// stable implementation: for a fast and stable implementation, November 2005
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::perturbCosts()
{
    original_costs = problem->costs;
    double magnitude;
    double perturbation;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    double min_perturb  = std::min(1e-2 * EPS_D, PSI);
    double max_perturb  = std::max(1e+3 * EPS_D, PSI * 10 * problem->costs.mean());

    for (int i = 0; i < problem->problem_size; i++)
    {
        magnitude = 100 * EPS_D + PSI * problem->costs[i];

        if (problem->bound_type[i] != BoundaryType::Lower && problem->bound_type[i] != BoundaryType::Free)
            perturbation = -0.5 * magnitude * (1 + dist(gen));
        perturbation= 0.5 * magnitude * (1 + dist(gen));
        
        perturbation = getWeight(problem->A.calcNonzeroInColumn(i)) * perturbation;

        while (fabs(perturbation) > fabs(max_perturb) || fabs(perturbation) < fabs(min_perturb))
        {
            if (fabs(perturbation) > fabs(max_perturb))
                perturbation = 0.1 * perturbation;
            else if (fabs(perturbation) < fabs(min_perturb))
                perturbation = 10 * perturbation;
        }

        problem->costs[i] += perturbation; 
    }

    perturbed = true;

    std::cout << "-- Cost perturbation done" << std::endl;
}


//----------------------------------------------------------------------------------------
// Weight for perturbation depends on two goals "keep nonzero count low" and 
// "resolver degeneracy"
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
double BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::getWeight(const int i) const
{
    std::vector<double> weights{0.01, 0.1, 1, 2, 5, 10, 20, 30, 40, 100};

    int k;
    if (i > 10)
        k = 9;
    else if (i == 0)
        k = 0;
    else
        k = i - 1;

    return weights[k];
}


//----------------------------------------------------------------------------------------
// Check primal feasibility in solver
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
bool BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::checkPrimalFeasible() const
{
    for (auto i : basis_indexes)
    {
        switch (problem->bound_type[i])
        {
        case BoundaryType::Fixed:
            if (fabs(x[i] - problem->lower_bound[i]) > EPS_P)
                return false;
            
        case BoundaryType::Free:
            break;

        case BoundaryType::Boxed:
            if (x[i] > problem->upper_bound[i] + EPS_R * problem->upper_bound[i] + EPS_P || 
                x[i] < problem->lower_bound[i] - EPS_R * problem->lower_bound[i] - EPS_P)
                return false;

        case BoundaryType::Upper:
            if (x[i] > problem->upper_bound[i] + EPS_R * problem->upper_bound[i] + EPS_P)
                return false;

        case BoundaryType::Lower:
            if (x[i] < problem->lower_bound[i] - EPS_R * problem->lower_bound[i] - EPS_P)
                return false;
        }
    }
    return true;
}



//----------------------------------------------------------------------------------------
// Check dual feasibility in solver
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
bool BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::checkDualFeasible() const
{
    for (auto i : non_basis_indexes)
    {
        switch (problem->bound_type[i])
        {
        case BoundaryType::Free:
            if (fabs(d[i]) < EPS_D || std::isnan(d[i]))
                return false;
            break;

        case BoundaryType::Upper:
            if (d[i] > EPS_D || std::isnan(d[i]))
                return false;
            break;

        case BoundaryType::Lower:
            if (d[i] < -EPS_D || std::isnan(d[i]))
                return false;
            break;
        }
    }
    return true;
}



//----------------------------------------------------------------------------------------
// Calc dual infeasibility 
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::calcDualInfeasible()
{
    obj_func_val = 0;
    for (auto i : non_basis_indexes)
    {
        switch (problem->bound_type[i])
        {
        case BoundaryType::Free:
            if (fabs(d[i]) < EPS_D || std::isnan(d[i]))
                obj_func_val += d[i];
            break;

        case BoundaryType::Upper:
            if (d[i] > EPS_D || std::isnan(d[i]))
                obj_func_val -= d[i];
            break;

        case BoundaryType::Lower:
            if (d[i] < -EPS_D || std::isnan(d[i]))
                obj_func_val += d[i];
            break;
        }
    }
}


//----------------------------------------------------------------------------------------
// COunt dual infeasibility 
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
int BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::counterDualInfeasible() const
{
    int num = 0;
    for (auto i : non_basis_indexes)
    {
        switch (problem->bound_type[i])
        {
        case BoundaryType::Free:
            if (fabs(d[i]) < EPS_D || std::isnan(d[i]))
                num += 1;
            break;

        case BoundaryType::Upper:
            if (d[i] > EPS_D || std::isnan(d[i]))
                num += 1;
            break;

        case BoundaryType::Lower:
            if (d[i] < -EPS_D || std::isnan(d[i]))
                num += 1;
            break;
        }
    }
    return num;
}


//----------------------------------------------------------------------------------------
// Calc infeasibility for j basis x
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
bool BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::setDelta(
    const int& j, double& delta, bool& is_lower
)
{
    if ((problem->bound_type[j] == BoundaryType::Boxed || 
        problem->bound_type[j] == BoundaryType::Lower ||
        problem->bound_type[j] == BoundaryType::Fixed ) && 
        problem->lower_bound[j] - x[j] > EPS_BOUND)
    {
        is_lower = true;
        delta = x[j] - problem->lower_bound[j]; 
        return true; 
    } 
    else if ((problem->bound_type[j] == BoundaryType::Boxed || 
                problem->bound_type[j] == BoundaryType::Upper ||
                problem->bound_type[j] == BoundaryType::Fixed ) && 
                x[j] - problem->upper_bound[j] > EPS_BOUND)
    {
        is_lower = false;
        delta = x[j] - problem->upper_bound[j];
        return true;
    }
    return false;
}


//----------------------------------------------------------------------------------------
// Choose candidates for ratio test aka CHUZR1
//----------------------------------------------------------------------------------------
template <typename MatrixType, typename VectorType, typename IndexVectorType>
bool BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::setRatioTestCandidates(
    IndexVector& F,const VectorType& tmp_alpha_p
)
{
    F.clear();
    for (int i = 0; i < non_basis_size; i++)
    {
        int j = non_basis_indexes[i];
        if ((tmp_alpha_p[i] > EPS_ALPHA && fabs(x[j] - problem->lower_bound[j]) < EPS_BOUND  &&
            (problem->bound_type[j] == BoundaryType::Lower || 
            problem->bound_type[j] == BoundaryType::Boxed)) ||
            (tmp_alpha_p[i] < -EPS_ALPHA && fabs(x[j] - problem->upper_bound[j]) < EPS_BOUND && 
            (problem->bound_type[j] == BoundaryType::Upper || 
            problem->bound_type[j] == BoundaryType::Boxed)) ||
            problem->bound_type[j] == BoundaryType::Free)
            F.push_back(i);
    }
    return (!F.size()) ? false : true;
}