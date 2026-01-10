#include "parallelDualSimplex.hpp"


bool parallelDualSimplex::minLex(const ValuesVector& a, const ValuesVector& b)
{
    size_t min_size = std::min(a.getSize(), b.getSize());
    
    for (size_t i = 0; i < min_size; ++i) {
        const double epsilon = 1e-12;
        if (std::fabs(a[i] - b[i]) > epsilon) 
            return a[i] < b[i];
        
    } 
    
    return a.getSize() < b.getSize();
}


ValuesVector parallelDualSimplex::prepareForLex(const ValuesVector& a, const size_t idx)
{
    size_t lex_size = a.getSize() + problem->logicals_size;
    ValuesVector lex_vector(lex_size);

    for (size_t i = 0; i < lex_size; i++)
        lex_vector[i] = (i < a.getSize()) ? a[i] : (i == idx + a.getSize()) ? 1 : 0;

    return lex_vector;
}


//----------------------------------------------------------------------------------------
// Convert string to exiting methods for presolver
//----------------------------------------------------------------------------------------
parallelDualSimplex::PresolverMethods parallelDualSimplex::stringToPreSolverMethod(
    const std::string& method_name
)
{
    static const std::unordered_map<std::string, PresolverMethods> methodMap = {
        {"minInfeas", PresolverMethods::minDualInfeasibility}
    };
    
    auto it = methodMap.find(method_name);
    if (it != methodMap.end()) {
        return it->second;
    }
    return PresolverMethods::UNKNOWN;
}


//----------------------------------------------------------------------------------------
// Convert string to exiting methods for solver
//----------------------------------------------------------------------------------------
parallelDualSimplex::SolverMethods parallelDualSimplex::stringToSolverMethod(
    const std::string& method_name
)
{
    static const std::unordered_map<std::string, SolverMethods> methodMap = {
        {"elaborated", SolverMethods::elaboratedMethod}
    };
    
    auto it = methodMap.find(method_name);
    if (it != methodMap.end()) {
        return it->second;
    }
    return SolverMethods::UNKNOWN;
}


//----------------------------------------------------------------------------------------
// Dual simplex method: Phase 1(Find dual feasible basis)
//----------------------------------------------------------------------------------------
parallelDualSimplex::Phase1OutStatus parallelDualSimplex::presolve(const std::string& presolver_method_name)
{
    auto method = stringToPreSolverMethod(presolver_method_name);
    if (checkPerturbNeed())
        perturbCosts();
    
    return callPresolver(method); 
}


//----------------------------------------------------------------------------------------
// Choose presolver 
//----------------------------------------------------------------------------------------
parallelDualSimplex::Phase1OutStatus parallelDualSimplex::callPresolver(const parallelDualSimplex::PresolverMethods method)
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


//----------------------------------------------------------------------------------------
// Choose solver 
//----------------------------------------------------------------------------------------
bool parallelDualSimplex::callDualSolver(const parallelDualSimplex::SolverMethods method)
{
    bool status;
    switch (method) 
    {
        case SolverMethods::elaboratedMethod:
            status = elaboratedMethod();
            break;

        case SolverMethods::UNKNOWN:
            throw "Unknown solver method";
            break;
    }
    return status;
}


//----------------------------------------------------------------------------------------
// Choose priaml solver 
//----------------------------------------------------------------------------------------
bool parallelDualSimplex::callPrimalSolver()
{
   return true;
}

//----------------------------------------------------------------------------------------
// Dual simplex method: Phase 2(Find solution)
//----------------------------------------------------------------------------------------
LPsolution parallelDualSimplex::solve(const std::string& method_name)
{
    std::cout << "Phase 2 : started" << std::endl;

    auto method = stringToSolverMethod(method_name);
    bool status_code = callDualSolver(method);
    
    if (status_code && perturbed) // optimal solution case
    {
        std::cout << "-- Optimal solution obtained with perturbation" << std::endl;

        problem->costs = original_costs;
        perturbed = false;

        if (!checkPrimalFeasible() && checkDualFeasible()) 
        {
            callDualSolver(method);

            std::cout << "-- Correction in primal infeasible and dual feasible case done" << std::endl;
        }
        else if (checkPrimalFeasible() && !checkDualFeasible())
        {
            callPrimalSolver();

            std::cout << "-- Correction in primal feasible and dual infeasible case done" << std::endl;
        }
        else if (!checkPrimalFeasible() && !checkDualFeasible())
        {
            Phase1OutStatus status_code = minimizeDualInfeasibility();
            if (status_code == Phase1OutStatus::Solved)
                callDualSolver(method);

            std::cout << "-- Correction in primal infeasible and dual infeasible case done" << std::endl;
        }
        std::cout << "-- Solution obtained" << std::endl;

        solution.x = x(0, problem->logicals_size);
        solution.Z = obj_func_val;

        std::cout << "Phase 2 : started" << std::endl;

        return solution;
    }
    else if (!status_code && perturbed) // dual unbound case
    {
        std::cout << "-- Optimal unbounded solution obtained with perturbation" << std::endl;

        problem->costs = original_costs;
        perturbed = false;

        Phase1OutStatus status_code = minimizeDualInfeasibility();
        if (status_code == Phase1OutStatus::Solved)
            callDualSolver(method);

        std::cout << "-- Correction done" << std::endl;
        std::cout << "-- Solution obtained" << std::endl;

        solution.x = x(0, problem->logicals_size);
        solution.Z = obj_func_val;

        std::cout << "Phase 2 : started" << std::endl;

        return solution;
    }
    else  // without perturbation case
    {
        std::cout << "-- Solution obtained" << std::endl;

        solution.x = x(0, problem->logicals_size);
        solution.Z = obj_func_val;

        std::cout << "Phase 2 : started" << std::endl;
        
        return solution;
    }
}


//----------------------------------------------------------------------------------------
// Constructor for solver, additionally finds dual feasible basis
//----------------------------------------------------------------------------------------
parallelDualSimplex::parallelDualSimplex(
    Problem& _problem, 
    const std::string& presolver_method_name
)
{
    std::cout << "Solver initialization : started" << std::endl;

    problem = &_problem;

    non_basis_size = problem->problem_size - problem->constraints_size;
    
    basis_indexes = IndexVector(problem->constraints_size);
    non_basis_indexes = IndexVector(non_basis_size);

    for (size_t i = 0; i < non_basis_size; i++)
        non_basis_indexes[i] = i;
    for (size_t i = 0; i < problem->constraints_size; i++)
        basis_indexes[i] = i + non_basis_size;

    std::cout << "Solver initialization : basis columns selected" << std::endl;

    x = ValuesVector(problem->problem_size);
    d = ValuesVector(problem->problem_size);    
    AN = problem->A(non_basis_indexes);
    B = problem->A(basis_indexes);
    linalg::PFIdecompose(B, B_eta_repr);

    std::cout << "Solver initialization : attributes setted" << std::endl;

    // if (!linalg::checkPFIdecompose(B_eta_repr, B)) std::cerr << "Incorrect decompose" << std::endl;
    std::cout << "Phase 1 : started" << std::endl;
    while(presolve(presolver_method_name) == Phase1OutStatus::NeedRestart);
    std::cout << "Phase 1 : ended" << std::endl;


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


//----------------------------------------------------------------------------------------
// Check if costs need perturbation to prevent cycling
//----------------------------------------------------------------------------------------
bool parallelDualSimplex::checkPerturbNeed() const
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
// Initialize weights for DSE according to
// John J. Forrest and Donald Goldfarb. Steepest-edge simplex algorithms for
// linear programming. Math. Program., 57(3):341–374, 1992.
//----------------------------------------------------------------------------------------
ValuesVector parallelDualSimplex::initBetaWeights()
{
    ValuesVector beta(problem->constraints_size);
    for (size_t i = 0; i < problem->constraints_size; i++)
        beta[i] = linalg::PFIsolve(B_eta_repr, linalg::unit(problem->constraints_size, i), true).norm();
    return beta;
}


//----------------------------------------------------------------------------------------
//Perturb costs according to 
// Dipl. Inform. Achim Koberstein. The Dual Simplex Method, Techniques for a fast and 
// stable implementation: for a fast and stable implementation, November 2005
//----------------------------------------------------------------------------------------
void parallelDualSimplex::perturbCosts()
{
    original_costs = problem->costs;
    double magnitude;
    double perturbation;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    double min_perturb  = std::min(1e-2 * EPS_D, PSI);
    double max_perturb  = std::max(1e+3 * EPS_D, PSI * 10 * problem->costs.mean());

    for (size_t i = 0; i < problem->problem_size; i++)
    {
        magnitude = 100 * EPS_D + PSI * problem->costs[i];

        if (problem->bound_type[i] != BoundaryType::Lower && problem->bound_type[i] != BoundaryType::Free)
            perturbation = -0.5 * magnitude * (1 + dist(gen));
        perturbation= 0.5 * magnitude * (1 + dist(gen));
        
        perturbation = getWeight(calcNonzeroInColumn(i)) * perturbation;

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
double parallelDualSimplex::getWeight(const size_t i) const
{
    std::vector<double> weights{0.01, 0.1, 1, 2, 5, 10, 20, 30, 40, 100};

    size_t k;
    if (i > 10)
        k = 9;
    else if (i == 0)
        k = 0;
    else
        k = i - 1;

    return weights[k];
}


//----------------------------------------------------------------------------------------
// Calc non zero elements in column of matrix A
//----------------------------------------------------------------------------------------
size_t parallelDualSimplex::calcNonzeroInColumn(const size_t i) const
{
    size_t non_zero_num = 0;
    for (auto a_j : problem->A(i))
        if (fabs(a_j) > EPS_A)
            non_zero_num += 1;
        
    return non_zero_num;
}


//----------------------------------------------------------------------------------------
// Check primal feasibility in solver
//----------------------------------------------------------------------------------------
bool parallelDualSimplex::checkPrimalFeasible() const
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
            if (x[i] - problem->upper_bound[i] > EPS_BOUND || problem->lower_bound[i] - x[i] > EPS_BOUND)
                return false;

        case BoundaryType::Upper:
            if (x[i] - problem->upper_bound[i] > EPS_BOUND)
                return false;

        case BoundaryType::Lower:
            if (problem->lower_bound[i] - x[i] > EPS_BOUND)
                return false;
        }
    }
    return true;
}



//----------------------------------------------------------------------------------------
// Check dual feasibility in solver
//----------------------------------------------------------------------------------------
bool parallelDualSimplex::checkDualFeasible() const
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
void parallelDualSimplex::calcDualInfeasible()
{
    obj_func_val = 0;
    for (auto i : non_basis_indexes)
    {
        switch (problem->bound_type[i])
        {
        case BoundaryType::Free:
            if (d[i] != 0)
                obj_func_val += d[i];
            break;

        case BoundaryType::Upper:
            if (d[i] > 0)
                obj_func_val -= d[i];
            break;

        case BoundaryType::Lower:
            if (d[i] < 0)
                obj_func_val += d[i];
            break;
        }
    }
}


//----------------------------------------------------------------------------------------
// COunt dual infeasibility 
//----------------------------------------------------------------------------------------
size_t parallelDualSimplex::counterDualInfeasible() const
{
    size_t num = 0;
    for (auto i : non_basis_indexes)
    {
        switch (problem->bound_type[i])
        {
        case BoundaryType::Free:
            if (fabs(d[i]) < EPS_D)
                num += 1;
            break;

        case BoundaryType::Upper:
            if (d[i] > EPS_D)
                num += 1;
            break;

        case BoundaryType::Lower:
            if (d[i] < -EPS_D)
                num += 1;
            break;
        }
    }
    return num;
}

//----------------------------------------------------------------------------------------
// Calc infeasibility for j basis x
//----------------------------------------------------------------------------------------
bool parallelDualSimplex::setDelta(const size_t& j, double& delta, bool& is_lower)
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
bool parallelDualSimplex::setRatioTestCandidates(IndexVector& F,const ValuesVector& tmp_alpha_p)
{
    F.clear();
    for (size_t i = 0; i < non_basis_size; i++)
    {
        size_t j = non_basis_indexes[i];
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


//----------------------------------------------------------------------------------------
// Phase 1 method for finding dual fesaible basis, based on article:
// E. Kostina. The long step rule in the bounded-variable dual simplex method:
// Numerical experiments. Mathematical Methods of Operations Research, 55:413–
// 429, 2002.
//----------------------------------------------------------------------------------------
parallelDualSimplex::Phase1OutStatus parallelDualSimplex::minimizeDualInfeasibility()
{
    // (Step 1) Initialization
    Phase1OutStatus status;
    ValuesVector y(problem->constraints_size);
    y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
    d.setValues(problem->costs(non_basis_indexes) - AN.T().dot(y), non_basis_indexes);

    IndexVector inf_u_indexes;
    IndexVector inf_l_indexes;
    IndexVector inf_f_indexes;
    for (size_t i = 0; i < non_basis_size; i++)
    {
        size_t j = non_basis_indexes[i];
        if ((problem->bound_type[j] == BoundaryType::Upper || 
            problem->bound_type[j] == BoundaryType::Free) && d[j] > EPS_D)
            inf_u_indexes.push_back(j);
        else if ((problem->bound_type[j] == BoundaryType::Lower || 
            problem->bound_type[j] == BoundaryType::Free) && d[j] < -EPS_D)
            inf_l_indexes.push_back(j);

        if (problem->bound_type[j] == BoundaryType::Free && d[j] < -EPS_D)
            inf_f_indexes.push_back(i);
    }
    
    obj_func_val = 0;
    ValuesVector columns_change(problem->constraints_size);
    for (auto i : inf_l_indexes)
    {
        obj_func_val += d[i];
        columns_change += problem->A(i);
    }

    for (auto i : inf_u_indexes)
    {
        obj_func_val += d[i];
        columns_change -= problem->A(i);
    }

    ValuesVector f(problem->constraints_size);
    f = linalg::PFIsolve(B_eta_repr, columns_change, false);

    ValuesVector beta(problem->constraints_size);
    beta = initBetaWeights();
    
    size_t iteration = 0;
    size_t cycle_num = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 1);
    std::uniform_int_distribution<> non_basis_rand(0, non_basis_size);

    std::unordered_set<size_t> blocked_p;
    while (true)
    {
        iteration += 1;
        if (!perturbed && cycle_num > MAX_CYCLE) 
        {
            perturbCosts();
            y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
            d.setValues(problem->costs(non_basis_indexes) - AN.T().dot(y), non_basis_indexes);
        }
        
        // (Step 2) Pricing
        size_t p, p_idx;
        double max_weight = 0;
        for (size_t i = 0; i < problem->constraints_size; i++)
        {
            size_t j = basis_indexes[i];
            if (((problem->bound_type[j] != BoundaryType::Free && problem->bound_type[j] != BoundaryType::Upper) && f[i] > EPS_BOUND) ||
                ((problem->bound_type[j] != BoundaryType::Free && problem->bound_type[j] != BoundaryType::Lower) && f[i] < -EPS_BOUND))
            {
                double weight_tmp = pow(f[i], 2) / beta[i];
                if (weight_tmp > max_weight && blocked_p.find(j) == blocked_p.end())
                {
                    p = j;
                    p_idx = i;
                    max_weight = weight_tmp;
                }
            }
            
        }

        if (checkDualFeasible())
        {
            calcDualInfeasible();
            if (obj_func_val == 0)
            {
                status = Phase1OutStatus::Solved;
            }
            else if (obj_func_val > 0)
            {
                solution.solved = false;
                solution.message = "dual infeasible";
                status = Phase1OutStatus::DualInfeas;
            }
            break;
        }
        
        // (Step 3) BTran
        ValuesVector rho(problem->constraints_size);
        rho = linalg::PFIsolve(B_eta_repr, linalg::unit(problem->constraints_size, p_idx), true);
        
        // (Step 4) Pivot row
         ValuesVector alpha(non_basis_size);
        alpha = AN.T().dot(rho);

        // (Step 5) Ratio Test
        ValuesVector alpha_tmp(non_basis_size);
        ValuesVector f_tmp(problem->constraints_size);
        if (f[p_idx] > 0)
        {
            alpha_tmp = -alpha;
            f_tmp = -f;
        }
        else
        {
            alpha_tmp = alpha;
            f_tmp = f;
        }
        
        IndexVector F, F_reserved;
        for (size_t i = 0; i < non_basis_indexes.size(); i++)
        {
            size_t j = non_basis_indexes[i];
            if (((d[j] >= 0 && alpha_tmp[i] > EPS_A) ||
                (d[j] <= 0 && alpha_tmp[i] < -EPS_A)))
                F.push_back(i);                
        }

        if (!F.size()) 
        {
            blocked_p.insert(p);
            if (blocked_p.find(p) == blocked_p.end())
            {
                status = Phase1OutStatus::NeedRestart;
                break;
            }
            continue;
        }

        if (blocked_p.size())
            blocked_p.clear();
        
        size_t q_idx = F[0];
        size_t q = non_basis_indexes[q_idx];;
        double theta = d[q] / alpha_tmp[q_idx];
        for (auto i : F)
        {   
            size_t j = non_basis_indexes[i];
            double theta_tmp = d[j] / alpha_tmp[i];
            if (theta_tmp < theta || (fabs(theta_tmp - theta) < EPS_Z && distrib(gen) == 1)) 
            {
                theta = theta_tmp;
                q = j;
                q_idx = i;
            } 
            else if (theta_tmp < theta + EPS_Z)
            {
                if (fabs(alpha_tmp[i]) > fabs(alpha_tmp[q_idx]))
                {
                    theta = theta_tmp;
                    q = j;
                    q_idx = i;
                }
            }
        }
        theta = d[q] / alpha[q_idx];
        
        // (Step 6) FTran
        ValuesVector alpha_q;
        alpha_q = linalg::PFIsolve(B_eta_repr, AN(q_idx), false);


        y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
        if (iteration % REFACT_FREQ == 0 || fabs(y.dot(AN(q_idx)) - problem->costs(basis_indexes).dot(alpha_q)) > REFACT_ERR)
        {
            B_eta_repr.clear();
            linalg::PFIdecompose(B, B_eta_repr);
        }

        // (Step 7) Basis change and update
        obj_func_val = obj_func_val - theta * f[p_idx];
        for (size_t i = 0; i < non_basis_indexes.size(); i++)
        {
            size_t j = non_basis_indexes[i];
            d[j] -= theta * alpha[i];    
        }
        d[p] = -theta;
        d[q] = 0;
        
        ValuesVector new_eta_matrix(problem->constraints_size);
        ValuesVector tau(problem->constraints_size);
        tau = linalg::PFIsolve(B_eta_repr, rho, false);
        for (size_t i = 0; i < problem->constraints_size; i++)
        {
            size_t j = basis_indexes[i];
            f[i] = (i != p_idx) ? f[i] - alpha_q[i] / alpha_q[p_idx] * f[p_idx] : f[p_idx];
            new_eta_matrix[i] = (i != p_idx) ? -alpha_q[i] / alpha_q[p_idx] : 1 / alpha_q[p_idx]; 
            beta[i] = (i != p_idx) ? beta[i] - 2 * alpha_q[i] / alpha_q[p_idx] * tau[i]  + pow(alpha_q[i] / alpha_q[p_idx], 2) * beta[p_idx] : beta[i]; 
        }  
        beta[p_idx] = beta[p_idx] / pow(alpha_q[p_idx], 2);
        f[p_idx] = f[p_idx] / alpha_q[p_idx];
        
        basis_indexes[p_idx] =  q;
        non_basis_indexes[q_idx] = p;
    
        B.swapColumn(AN, p_idx, q_idx);
        B_eta_repr.push_back(EtaMatrix(new_eta_matrix, p_idx));

        inf_f_indexes.clear();
        for (size_t i = 0; i < non_basis_size; i++)
        {
            size_t j = non_basis_indexes[i];
            if (problem->bound_type[j] == BoundaryType::Free && d[j] < 0)
                inf_f_indexes.push_back(i);
        }

        calcDualInfeasible();
        if (fabs(theta) < EPS_A) cycle_num += 1;

        #ifdef DEBUG
        std::cout << iteration << " : Z = "<< obj_func_val << " p = " << p << " q = " << q << " inf_num = " << counterDualInfeasible() << std::endl;  
        #endif

        // if (!linalg::checkPFIdecompose(B_eta_repr, B)) std::cerr << "Incorrect decompose" << std::endl;
    }
    return status;
}


//----------------------------------------------------------------------------------------
// Phase 2 simple method for finding solution, based on article:
// Dipl. Inform. Achim Koberstein. The Dual Simplex Method, Techniques for a fast and 
// stable implementation: for a fast and stable implementation, November 2005
//----------------------------------------------------------------------------------------
bool parallelDualSimplex::elaboratedMethod()
{
    auto start = std::chrono::high_resolution_clock::now();

    // (Step 1) Initialization
    ValuesVector y(problem->constraints_size);
    ValuesVector beta(problem->constraints_size);

    problem->RHS = problem->RHS - AN.dot(x(non_basis_indexes));
    x.setValues(linalg::PFIsolve(B_eta_repr, problem->RHS, false), basis_indexes);
    y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
    d.setValues(problem->costs(non_basis_indexes) - AN.T().dot(y), non_basis_indexes);
    obj_func_val = problem->costs.dot(x);
    beta = initBetaWeights();
    size_t iteration = 0;
    size_t cycle_num = 0;


    while (true)
    {
        if (!perturbed && cycle_num > MAX_CYCLE) 
        {
            perturbCosts();
            y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
            d.setValues(problem->costs(non_basis_indexes) - AN.T().dot(y), non_basis_indexes);
            obj_func_val = problem->costs.dot(x);
        }

        #ifdef DEBUG
            std::cout << iteration << " : Z = "<< obj_func_val << std::endl;
        #endif

        // (Step 2) Pricing
        if (checkPrimalFeasible())
        {
            solution.solved = true;
            solution.message = "optimal solution";
            break;
        }
        
        size_t max_candidates_num = omp_get_num_threads();
        IndexVector pivot_candidats(max_candidates_num);
        ValuesVector candidate_weights(max_candidates_num);
        for (size_t i = 0; i < problem->constraints_size; i++)
        {
            size_t j = basis_indexes[i];
            double delta_tmp = 0;
            bool is_lower_tmp; 
           
            bool _ = setDelta(j, delta_tmp, is_lower_tmp);

            double weight_tmp = pow(delta_tmp, 2) / beta[i];
            for (size_t k = 0; k < max_candidates_num; k++)
            {
                if (weight_tmp > candidate_weights[k] && delta_tmp != 0)
                {
                    pivot_candidats[k] = i;
                    candidate_weights[k] = weight_tmp;
                    break;
                }
                
            }
        }

        size_t candidates_num = candidate_weights.countNonZero();
        size_t dual_unbound_counter = 0;
        std::vector<double> Z_deltas(candidates_num);
        IndexVector entering_candidates(candidates_num);
        std::vector<ValuesVector> candidate_betas(candidates_num);
        std::vector<ValuesVector> candidate_x(candidates_num);
        std::vector<ValuesVector> candidate_d(candidates_num);

        #pragma omp parallel for
        for (size_t cand_id = 0; cand_id < candidates_num; cand_id++)
        {
            size_t p_idx = pivot_candidats[cand_id];
            size_t p = basis_indexes[p_idx];
            bool is_lower;
            double delta;
            bool _ = setDelta(p, delta, is_lower);
            
            // (Step 3) BTran
            ValuesVector rho(problem->constraints_size);
            rho = linalg::PFIsolve(B_eta_repr, linalg::unit(problem->constraints_size, p_idx), true);

            // (Step 4) Pivot row
            ValuesVector alpha_p(non_basis_size);
            alpha_p = AN.T().dot(rho);

            // (Step 5) Ratio Test
            ValuesVector tmp_alpha_p(non_basis_size);
            delta = fabs(delta);
            int sgn = (is_lower) ? -1 : 1;
            tmp_alpha_p = (is_lower) ? -alpha_p : alpha_p;

            IndexVector F;
            bool F_not_empty = setRatioTestCandidates(F, tmp_alpha_p);
            if (!F_not_empty)
            {
                dual_unbound_counter += 1;
                continue;
            }

            size_t q_idx, q;
            double theta;
            while (F.size() && delta >= 0)
            {
                size_t it = 0;
                q_idx = F[0];
                q = non_basis_indexes[q_idx];
                theta = (d[q] / tmp_alpha_p[q_idx]);
                for (auto i : F)
                {   
                    size_t j = non_basis_indexes[i];
                    double theta_tmp = d[j] / tmp_alpha_p[i];
                    if (theta_tmp < theta || (theta_tmp == theta && minLex(prepareForLex(AN(i), j), prepareForLex(AN(q_idx), q))))
                    {
                        theta = theta_tmp;
                        q = j;
                        q_idx = i;
                    }
                }
                if (problem->bound_type[q] == BoundaryType::Boxed)
                {
                    delta -= (problem->upper_bound[q] - problem->lower_bound[q]) * fabs(alpha_p[q_idx]);
                    auto _ = std::remove(F.begin(), F.end(), q);
                }
                else
                {
                    break;
                }
            }
            theta = d[q] / alpha_p[q_idx];
            delta = sgn * delta;

            if (!F.size())
            {
                dual_unbound_counter += 1;
                continue;
            }

            // (Step 6) FTran
            ValuesVector alpha_q;
            alpha_q = linalg::PFIsolve(B_eta_repr, AN(q_idx), false);

            // (Step 7) Basis change and update

            // Update d accoprding to BRFT
            ValuesVector column_change(problem->constraints_size);
            IndexVector infeas_idx;
            double delta_z = 0;
            ValuesVector updated_d = d;
            for (size_t i = 0; i < non_basis_indexes.size(); i++)
            {
                size_t j = non_basis_indexes[i];
                updated_d[j] = d[j] - theta * alpha_p[i];  
                if (problem->bound_type[j] == BoundaryType::Boxed)
                {
                
                    if (x[j] == problem->lower_bound[j] && updated_d[j] < 0)
                    {
                        infeas_idx.push_back(j);
                        column_change += AN(i) * (problem->upper_bound[j] - problem->lower_bound[j]);
                        delta_z += problem->costs[j] * (problem->upper_bound[j] - problem->lower_bound[j]);
                    }  
                    else if (x[j] == problem->upper_bound[j] && updated_d[j] > 0)
                    {
                        infeas_idx.push_back(j);
                        column_change -= AN(i) * (problem->upper_bound[j] - problem->lower_bound[j]);
                        delta_z -= problem->costs[j] * (problem->upper_bound[j] - problem->lower_bound[j]);
                    }   
                }          
            }
            updated_d[p] = -theta;
            updated_d[q] = 0;

            ValuesVector delta_xB;
            ValuesVector updated_x = x;
            if (!infeas_idx.size())
            {
                delta_xB = linalg::PFIsolve(B_eta_repr, column_change, false);
                updated_x.setValues(updated_x(basis_indexes) - delta_xB, basis_indexes);
                for (size_t i = 0; i < problem->constraints_size; i++)
                {
                    size_t j = basis_indexes[i];
                    delta_z -= problem->costs[j] * delta_xB[i];
                }    
            }

            // Update xB, DSE weights
            double theta_P = delta / alpha_q[p_idx];
            ValuesVector tau(problem->constraints_size);
            ValuesVector updated_beta = beta;
            tau = linalg::PFIsolve(B_eta_repr, rho, false);
            for (size_t i = 0; i < problem->constraints_size; i++)
            {
                size_t j = basis_indexes[i];
                updated_x[j] = x[j] - theta_P * alpha_q[i];  
                updated_beta[i] = (i != p_idx) ? beta[i] - 2 * alpha_q[i] / alpha_q[p_idx] * tau[i]  + pow(alpha_q[i] / alpha_q[p_idx], 2) * beta[p_idx] : beta[i]; 
            }  
            updated_beta[p_idx] = beta[p_idx] / pow(alpha_q[p_idx], 2);
            updated_x[q] = x[q] + theta_P;

            double new_delta = 0;
            bool is_infeasible = setDelta(p, new_delta, _);
            double new_weight = pow(new_delta, 2) / beta[p_idx];
            if (!is_infeasible || new_weight < CAND_RATIO * candidate_weights[cand_id])
                continue;
            
            // Flip bounds
            for (auto j : infeas_idx)
            {
                if (x[j] == problem->lower_bound[j])
                    updated_x[j] = problem->upper_bound[j];
                else 
                    updated_x[j] = problem->lower_bound[j]; 
            }

            entering_candidates[cand_id] = q_idx;
            candidate_betas[cand_id] = updated_beta;
            candidate_x[cand_id] = updated_x;
            candidate_d[cand_id] = updated_d;
            Z_deltas[cand_id] = (theta * delta + delta_z);
        }

        if (dual_unbound_counter == candidates_num)
        {
            solution.solved = false;
            solution.message = "dual unbounded";
            break;
        }
        

        size_t opt_cand_id = std::distance(Z_deltas.begin(), std::max_element(Z_deltas.begin(), Z_deltas.end()));
        size_t opt_p_idx   = pivot_candidats[opt_cand_id];
        size_t opt_p       = basis_indexes[opt_p_idx];
        size_t opt_q_idx   = entering_candidates[opt_cand_id];
        size_t opt_q       = non_basis_indexes[opt_q_idx];
        x =    candidate_x[opt_cand_id];
        beta = candidate_betas[opt_cand_id];
        d =    candidate_d[opt_cand_id];

        // Update basis
        ValuesVector opt_alpha_q;
        opt_alpha_q = linalg::PFIsolve(B_eta_repr, AN(opt_q_idx), false);

        ValuesVector new_eta_matrix(problem->constraints_size);
        for (size_t i = 0; i < problem->constraints_size; i++)
            new_eta_matrix[i] = (i != opt_p_idx) ? -opt_alpha_q[i] / opt_alpha_q[opt_p_idx] : 1 / opt_alpha_q[opt_p_idx]; 
        
        basis_indexes[opt_p_idx] =  opt_q;
        non_basis_indexes[opt_q_idx] = opt_p;
        B.swapColumn(AN, opt_p_idx, opt_q_idx);
        B_eta_repr.push_back(EtaMatrix(new_eta_matrix, opt_p_idx));

        obj_func_val += Z_deltas[opt_cand_id];
        iteration += 1;
        if (fabs(Z_deltas[opt_cand_id]) < EPS_A) cycle_num += 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time spent: " << duration_ns.count() << " ms\n";
    return solution.solved;
}


