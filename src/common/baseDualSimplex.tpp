#include "baseDualSimplex.hpp"


template <typename MatrixType, typename VectorType, typename IndexVectorType>
Phase1OutStatus BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::presolve(const std::string& presolver_method_name)
{
    std::cout << "Phase 1 : started" << std::endl;
    auto method = stringToPreSolverMethod(presolver_method_name);
    presolver_method = method;

    if (checkPerturbNeed())
        perturbCosts();

    initBetaWeights(true);   

    Phase1OutStatus status = callPresolver(method);
    
    if (status == Phase1OutStatus::DualInfeas)
    {
        std::cout << "-- Problem is dual infeasible" << std::endl;
        return status;
    }
    std::cout << "-- Problem is feasible" << std::endl;

    setPrimalVars();
    
    std::cout << "-- Nonbasis variables bounds setted done" << std::endl;
    std::cout << "Phase 1 : ended" << std::endl;
    return status;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::initBetaWeights(bool Ibasis)
{
    if (Ibasis)
    {
        for (int i = 0; i < basis_size; i++)
            beta[i] = 1;   
    }
    else
    {
        if constexpr (std::is_same_v<VectorType, ValuesVector>) 
        {
            VectorType buff(basis_size);
            for (int i = 0; i < basis_size; i++)
            {
                VectorType unit_vec(linalg::unit(problem->constraints_size, i));
                solveLinSys(unit_vec, buff, true);
                beta[i] = buff.norm();
            }
        }
        else
        {
            VectorType buff(basis_size);
            for (int i = 0; i < basis_size; i++)
            {
                VectorType unit_vec(linalg::unit(problem->constraints_size, i));
                solveLinSys(unit_vec, buff, true);
                buff.updateHostMem();
                // buff.show();
                beta[i] = buff.norm();
                // std::cout << beta[i] << std::endl;
                //  exit(0);

            }
        }
    }
    
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::setPrimalVars()
{
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
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
Phase1OutStatus BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::callPresolver(const PresolverMethods method)
{
    Phase1OutStatus status;
    switch (method) 
    {
        case PresolverMethods::minDualInfeasibility:
            status = minimizeDualInfeasibility();
            break;
        
        case PresolverMethods::UNKNOWN:
            throw "Unknown phase 1 solver method";
            break;
    }
    return status;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::postProcess(bool status_code)
{
    if (perturbed)
    {   
        problem->costs = original_costs;
        dualSimplexInit();

        if (status_code)
        {
            std::cout << "-- Optimal solution obtained with perturbation" << std::endl;

            if (!checkDualFeasible()) 
            {
                callPrimalSolver();

                std::cout << "-- Correction in primal feasible and dual infeasible case done" << std::endl;
            }
            std::cout << "-- Solution obtained" << std::endl;
            problem->solution.x = x;
            problem->solution.Z += obj_func_val;
        }
        else
        {
            std::cout << "-- Unbounded solution obtained with perturbation" << std::endl;
           
            problem->solution.x = x;
            problem->solution.Z += obj_func_val;
        }
    }
    else
    {
        std::cout << "-- Solution obtained" << std::endl;

        problem->solution.x = x;
        problem->solution.Z += obj_func_val;
    }
}



template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::solve()
{
    std::cout << "Phase 2 : started" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    bool status_code = callDualSolver();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
   
    std::cout << "time = " << duration_ms / 1000 << " s" << std::endl;

    postProcess(status_code);

    std::cout << "Phase 2 : ended" << std::endl;
}


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
    
    beta                = VectorType(basis_size);

    for (int i = 0; i < non_basis_size; i++)
        non_basis_indexes[i] = i;
    
    for (int i = 0; i < basis_size; i++)
        basis_indexes[i] = i + non_basis_size;
}


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


template <typename MatrixType, typename VectorType, typename IndexVectorType>
bool BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::checkPerturbNeed() const
{
    std::vector<double> costs(problem->costs.begin(), problem->costs.end());
    std::sort(costs.begin(), costs.end());

    auto last = std::unique(costs.begin(), costs.end(),
        [](double a, double b) { return std::fabs(a - b) < EPS_COSTS; });

    int unique_count = std::distance(costs.begin(), last);

    return unique_count < PERTURB_RATIO * problem->costs.getSize();
}


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


template <typename MatrixType, typename VectorType, typename IndexVectorType>
bool BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::checkPrimalFeasible() const
{
    for (auto i : basis_indexes)
    {
        switch (problem->bound_type[i])
        {
        case BoundaryType::Fixed:
            if (fabs(x[i] - problem->lower_bound[i]) > EPS_BOUND)
                return false;
            
        case BoundaryType::Free:
            break;

        case BoundaryType::Boxed:
            if (x[i] > problem->upper_bound[i] + EPS_R * fabs(problem->upper_bound[i]) + EPS_BOUND || 
                x[i] < problem->lower_bound[i] - EPS_R * fabs(problem->lower_bound[i]) - EPS_BOUND)
                return false;

        case BoundaryType::Upper:
            if (x[i] > problem->upper_bound[i] + EPS_R * fabs(problem->upper_bound[i]) + EPS_BOUND)
                return false;

        case BoundaryType::Lower:
            if (x[i] < problem->lower_bound[i] - EPS_R * fabs(problem->lower_bound[i]) - EPS_BOUND)
                return false;
        }
    }
    return true;
}



template <typename MatrixType, typename VectorType, typename IndexVectorType>
int BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::counterPrimalInfeasible() const
{
    int num = 0;
    for (auto i : basis_indexes)
    {
        switch (problem->bound_type[i])
        {
        case BoundaryType::Fixed:
            if (fabs(x[i] - problem->lower_bound[i]) > EPS_BOUND)
                num++;
            
        case BoundaryType::Free:
            break;

        case BoundaryType::Boxed:
            if (x[i] > problem->upper_bound[i] + EPS_R * fabs(problem->upper_bound[i]) + EPS_BOUND || 
                x[i] < problem->lower_bound[i] - EPS_R * fabs(problem->lower_bound[i]) - EPS_BOUND)
                num++;

        case BoundaryType::Upper:
            if (x[i] > problem->upper_bound[i] + EPS_R * fabs(problem->upper_bound[i]) + EPS_BOUND)
                num++;

        case BoundaryType::Lower:
            if (x[i] < problem->lower_bound[i] - EPS_R * fabs(problem->lower_bound[i]) - EPS_BOUND)
                num++;
        }
    }
    return num;
}


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

template <typename MatrixType, typename VectorType, typename IndexVectorType>
std::pair<bool, double> BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::checkVarIsInfeas(int i)
{
    switch (problem->bound_type[i])
    {
    case BoundaryType::Lower:
        if (x[i] < problem->lower_bound[i] - EPS_BOUND)
            return std::make_pair(true, x[i] - problem->lower_bound[i]);

    case BoundaryType::Upper:
        if (x[i] > problem->upper_bound[i] + EPS_BOUND)
            return std::make_pair(true, x[i] - problem->upper_bound[i]);

    case BoundaryType::Fixed:
        if (fabs(x[i] - problem->lower_bound[i]) < EPS_BOUND)
            return std::make_pair(true, x[i] - problem->lower_bound[i]);

    case BoundaryType::Boxed:
        if (x[i] > problem->upper_bound[i] + EPS_BOUND)
            return std::make_pair(true, x[i] - problem->upper_bound[i]);
        else if (x[i] < problem->lower_bound[i] - EPS_BOUND)
            return std::make_pair(true, x[i] - problem->lower_bound[i]);
    }
    return std::make_pair(false, 0);
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
typename BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::PricingInfo BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::DSEPricing(std::unordered_set<int>& blocked_p)
{
    double delta;
    int p = -1;
    int p_idx = -1;
    bool is_lower = false;
    double max_weight = 0;

    for (int i = 0; i < basis_size; i++)
    {
        int j = basis_indexes[i];
        bool is_lower_tmp; 
        auto [is_infeas, delta_tmp] = checkVarIsInfeas(j);

        if (is_infeas && fabs(delta_tmp) > 1e-8)
        {
            is_lower_tmp = delta_tmp < 0;
            double weight_tmp = pow(delta_tmp, 2) / beta[i];

            if (weight_tmp > max_weight && blocked_p.find(j) == blocked_p.end())
            {
                p = j;
                p_idx = i;
                delta  = delta_tmp;
                max_weight = weight_tmp;
                is_lower = is_lower_tmp;
            }
        }
    }

    return std::make_tuple(p, p_idx, delta, is_lower);
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
typename BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::PricingInfo BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::simplePricing(std::unordered_set<int>& blocked_p)
{
    double delta;
    int p, p_idx;
    bool is_lower = false;

    for (int i = 0; i < basis_size; i++)
    {
        int j = basis_indexes[i];
        
        auto [is_infeas, delta_tmp] = checkVarIsInfeas(j);

        if (is_infeas && blocked_p.find(j) == blocked_p.end())
        {
            p = j;
            p_idx = i;
            is_lower = delta_tmp < 0;
            delta = delta_tmp;
            break;
        }
    }
    return std::make_tuple(p, p_idx, delta, is_lower);
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
IndexVector BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::phase1SetRatioTestCandidates(const VectorType& alpha)
{
    IndexVector F;
    for (int i = 0; i < non_basis_size; i++)
    {
        int j = non_basis_indexes[i];
        if (((d[j] >= 0 && alpha[i] > EPS_A) ||
            (d[j] <= 0 && alpha[i] < -EPS_A)))
            F.push_back(i);                
    }
    return F;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
typename BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::RatioTestInfo BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::simpleRatioTest(IndexVector& F, const VectorType& alpha)
{
    int q_idx, q;
    double theta = INF;
    for (auto i : F)
    {   
        int j = non_basis_indexes[i];
        double theta_tmp = d[j] / alpha[i];
        if (theta_tmp < theta) 
        {
            theta = theta_tmp;
            q = j;
            q_idx = i;
        } 
        else if (fabs(theta_tmp - theta) < EPS_Z)
        {
            if (fabs(alpha[i]) > fabs(alpha[q_idx]))
            {
                theta = theta_tmp;
                q = j;
                q_idx = i;
            }
        }
    }

    return std::make_tuple(q, q_idx);
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
typename BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::RatioTestInfo BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::harrisRatioTest(IndexVector& F, IndexVector& F_l, IndexVector& F_u, const VectorType& alpha)
{
    int q_idx = -1, q = -1;
    double theta_max = INF;
    double min_theta_l = INF;
    double min_theta_u = INF;

    for (auto i : F_l)
    {
        int j = non_basis_indexes[i];
        min_theta_l = std::min((d[j] + EPS_D) / alpha[i], min_theta_l);
    }

    for (auto i : F_u)
     {
        int j = non_basis_indexes[i];
        min_theta_u = std::min((d[j] - EPS_D) / alpha[i], min_theta_u);
    }

    theta_max = std::min(min_theta_l, min_theta_u);

    double alpha_max = 0;
    for (auto i : F)
    {   
        int j = non_basis_indexes[i];
        double theta_tmp = d[j] / alpha[i];
        if (theta_tmp <= theta_max && fabs(alpha[i]) >= alpha_max) 
        {
            alpha_max = fabs(alpha[i]);
            q = j;
            q_idx = i;
        } 
    }

    return std::make_tuple(q, q_idx);
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
double BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::infisibilityCorr(int q)
{
    double infisib_corr = 0;
    if ((problem->bound_type[q] == BoundaryType::Upper || 
        problem->bound_type[q] == BoundaryType::Free) && d[q] > EPS_D)  
        infisib_corr = 1;
    else if ((problem->bound_type[q] == BoundaryType::Lower || 
        problem->bound_type[q] == BoundaryType::Free) && d[q] < -EPS_D) 
        infisib_corr = -1;
    return infisib_corr;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::simpleReducedCostsUpate(const VectorType& alpha, int p, int q, double theta)
{
    return;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
PresolverMethods BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::stringToPreSolverMethod(
    const std::string& method_name
)
{
    return PresolverMethods::UNKNOWN;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::solveLinSys(
    VectorType&& rhs, VectorType& sol, bool transpose
)
{
    return;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::dualSimplexInit()
{
    return;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::solveLinSys(
    VectorType& rhs, VectorType& sol, bool transpose
)
{
    return;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
bool BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::callDualSolver()
{
    return true;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
bool BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::callPrimalSolver()
{
   return true;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::BTran(int p_idx, VectorType& rho)
{
    return;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::pivotRow(VectorType& rho, VectorType& alpha)
{
    return;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::FTran(int q, VectorType& alpha_q)
{
    return;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::initReducedCosts()
{
    return;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::initPhase1PricingVector(VectorType& f, IndexVector& inf_u_indexes, IndexVector& inf_l_indexes, double& Z)
{
    return;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::reFactorize()
{
    return;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::updateAndChangeBasis(
    VectorType& f, VectorType& rho, const VectorType& alpha_q, 
    int p_idx, int p, int q_idx, int q, double theta_P
)
{
    return;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::phase1UpdateAndChangeBasis(
    VectorType& f, VectorType& rho, const VectorType& alpha_q, 
    int p_idx, int p, int q_idx, int q, double theta_P
)
{
    return;
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
typename BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::Phase1PricingInfo BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::phase1Pricing(VectorType& f, std::unordered_set<int>& blocked_p)
{
    double max_weight = 0;
    double weight_tmp;
    bool candid_find = false;
    int p = -1;
    int p_idx = -1;
    for (int i = 0; i < basis_size; i++)
    {
        int j = basis_indexes[i];
        if ((problem->bound_type[j] == BoundaryType::Lower && f[i] > EPS_F) ||
            (problem->bound_type[j] == BoundaryType::Upper && f[i] < -EPS_F) ||
            problem->bound_type[j] == BoundaryType::Boxed ||
            problem->bound_type[j] == BoundaryType::Fixed)
        {
            weight_tmp = pow(f[i], 2) / beta[i];
            if (weight_tmp > max_weight && blocked_p.find(j) == blocked_p.end())
            {
                p = j;
                p_idx = i;
                max_weight = weight_tmp;
                candid_find = true;
            }
        }
    }
    return std::make_tuple(p, p_idx, candid_find);
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
void BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::minimizeDualInfeasibilityInit(VectorType& f)
{
    IndexVector inf_u_indexes;
    IndexVector inf_l_indexes;

    initReducedCosts();

    for (int i = 0; i < non_basis_size; i++)
    {
        int j = non_basis_indexes[i];
        if ((problem->bound_type[j] == BoundaryType::Upper || 
            problem->bound_type[j] == BoundaryType::Free) && d[j] > EPS_D)
            inf_u_indexes.push_back(j);
        else if ((problem->bound_type[j] == BoundaryType::Lower || 
            problem->bound_type[j] == BoundaryType::Free) && d[j] < -EPS_D)
            inf_l_indexes.push_back(j);
    }

    initPhase1PricingVector(f, inf_u_indexes, inf_l_indexes, obj_func_val);
}


template <typename MatrixType, typename VectorType, typename IndexVectorType>
Phase1OutStatus BaseDualSimplex<MatrixType, VectorType, IndexVectorType>::minimizeDualInfeasibility()
{
    // (Step 1) Initialization
    Phase1OutStatus status;

    VectorType rho(basis_size);                   // basis_size
    VectorType f(basis_size);                     // basis_size
    VectorType alpha(non_basis_size);             // non_basis_size
    VectorType alpha_q(basis_size);               // basis_size
    
    minimizeDualInfeasibilityInit(f);
    
    int cycle_num = 0;

    #ifdef DEBUG
        std::cout << iteration << " : Z = "<< obj_func_val << " inf_num = " << counterDualInfeasible() << std::endl;  
    #endif
    std::unordered_set<int> blocked_p;
    while (true)
    {
        iteration += 1;

        if (!perturbed && cycle_num > MAX_CYCLE) 
        {
            perturbCosts();
            minimizeDualInfeasibilityInit(f);
        }

        if (cycle_num > NEED_RESTART || 
            obj_func_val > INF || std::isnan(obj_func_val))
        {
            minimizeDualInfeasibilityInit(f);
        }

        if (iteration % REFACT_FREQ == 0)
        {
            reFactorize();
        }

        // (Step 2) Pricing
        auto [p, p_idx, candid_find] = phase1Pricing(f, blocked_p);

        if (!candid_find || !counterDualInfeasible())
        {
            if (fabs(obj_func_val) < EPS_A || !counterDualInfeasible())
            {
                status = Phase1OutStatus::Solved;
                break;
            }
            else if (obj_func_val > 0)
            {
                problem->solution.solved = false;
                problem->solution.message = "dual infeasible";
                status = Phase1OutStatus::DualInfeas;
                break;
            }
        }

        if (p < 0 || p_idx < 0)
        {
            std::cout << "-- Soft restart min infeas. No p find." << std::endl;  
            reFactorize();
            minimizeDualInfeasibilityInit(f);
            initBetaWeights(false);
            blocked_p.clear();
            continue;
        }

        if (obj_func_val > EPS_BOUND)
        {
            std::cout << "-- Soft restart min infeas. Z > 0." << std::endl; 
            reFactorize();
            minimizeDualInfeasibilityInit(f);
            initBetaWeights(false);
            blocked_p.clear();
            continue;
        }

        // (Step 3) BTran
        BTran(p_idx, rho);
        
        // (Step 4) Pivot row
        pivotRow(rho, alpha);

        // (Step 5) Ratio Test
        if (f[p_idx] > 0) alpha.multiplyData(-1);
        IndexVector F = std::move(phase1SetRatioTestCandidates(alpha));
        
        // rty to save
        if (!F.size()) 
        {
            #ifdef DEBUG
                std::cout << "-- Soft restart min infeas." << std::endl;  
            #endif

            blocked_p.insert(p);

            reFactorize();
            minimizeDualInfeasibilityInit(f);

            continue;
        }
        // end trying

        if (blocked_p.size())
            blocked_p.clear();

        auto [q, q_idx] = simpleRatioTest(F, alpha);
        if (f[p_idx] > 0) alpha.multiplyData(-1);
        double theta = d[q] / alpha[q_idx];
        
        // (Step 6) FTran
        FTran(q, alpha_q);
        if (fabs(alpha_q[p_idx] - alpha[q_idx]) > EPS_R * (1 + fabs(alpha_q[p_idx])))
        {
            reFactorize();
        }

        // (Step 7) Basis change and update
        double infisib_corr = infisibilityCorr(q);
        
        obj_func_val = obj_func_val - theta * f[p_idx];
        simpleReducedCostsUpate(alpha, p, q, theta);

        double theta_P = f[p_idx] / alpha_q[p_idx];
        phase1UpdateAndChangeBasis(f, rho, alpha_q, p_idx, p, q_idx, q, theta_P);
        f[p_idx] = f[p_idx] / alpha_q[p_idx] + infisib_corr;

        cycle_num = (fabs(theta) < EPS_A) ? cycle_num + 1 : 0;

        #ifdef DEBUG
            std::cout << iteration << " : Z = "<< obj_func_val << " inf_num = " << counterDualInfeasible() << " p_info:" << p << " q_info:" << q << " f:" << f[p_idx] << " theta:" << theta << " alpha_q:" << alpha_q[p_idx] << std::endl;  
        #endif
    }
    return status;
}
