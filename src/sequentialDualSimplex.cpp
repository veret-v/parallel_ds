#include "sequentialDualSimplex.hpp"


void LPsolution::show()
{
    std::cout << "-----------------------------LP result-----------------------------" << std::endl;
    std::cout << "solved:                     " << solved << std::endl;
    std::cout << "status:                     " << message << std::endl;
    std::cout << "Z(obj. func. value):        " << Z << std::endl;
    std::cout << "x0:                         " << x[0] << std::endl;
    for (size_t i = 1; i < std::min(int(x.getSize()), 10); i++)
        std::cout << "                            " <<  x[i] << std::endl;
    if (x.getSize() > 10)
        std::cout << "                            " <<  "..." << std::endl;
    
}



bool sequentialDualSimplex::minLex(const ValuesVector& a, const ValuesVector& b)
{
    size_t min_size = std::min(a.getSize(), b.getSize());
    
    for (size_t i = 0; i < min_size; ++i) {
        const double epsilon = 1e-12;
        if (std::fabs(a[i] - b[i]) > epsilon) 
            return a[i] < b[i];
        
    } 
    
    return a.getSize() < b.getSize();
}


ValuesVector sequentialDualSimplex::prepareForLex(const ValuesVector& a, const size_t idx)
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
sequentialDualSimplex::PresolverMethods sequentialDualSimplex::stringToPreSolverMethod(
    const std::string& method_name
)
{
    static const std::unordered_map<std::string, PresolverMethods> methodMap = {
        {"minInfeas", PresolverMethods::minDualInfeasibility},
        {"Pan", PresolverMethods::panMethod}
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
sequentialDualSimplex::SolverMethods sequentialDualSimplex::stringToSolverMethod(
    const std::string& method_name
)
{
    static const std::unordered_map<std::string, SolverMethods> methodMap = {
        {"simple", SolverMethods::simpleRatio},
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
sequentialDualSimplex::Phase1OutStatus sequentialDualSimplex::presolve(const std::string& presolver_method_name)
{
    auto method = stringToPreSolverMethod(presolver_method_name);
    if (checkPerturbNeed())
        perturbCosts();
    
    return callPresolver(method); 
}


//----------------------------------------------------------------------------------------
// Choose presolver 
//----------------------------------------------------------------------------------------
sequentialDualSimplex::Phase1OutStatus sequentialDualSimplex::callPresolver(const sequentialDualSimplex::PresolverMethods method)
{
    Phase1OutStatus status;
    switch (method) 
    {
        case PresolverMethods::minDualInfeasibility:
            status = minimizeDualInfeasibility();
            break;

        case PresolverMethods::panMethod:
            status = panMathod();
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
bool sequentialDualSimplex::callDualSolver(const sequentialDualSimplex::SolverMethods method)
{
    bool status;
    switch (method) 
    {
        case SolverMethods::simpleRatio:
            status = simpleRatioMethod();
            break;

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
bool sequentialDualSimplex::callPrimalSolver()
{
   return true;
}

//----------------------------------------------------------------------------------------
// Dual simplex method: Phase 2(Find solution)
//----------------------------------------------------------------------------------------
LPsolution sequentialDualSimplex::solve(const std::string& method_name)
{
    auto method = stringToSolverMethod(method_name);
    bool status_code = callDualSolver(method);
    
    if (status_code && perturbed) // optimal solution case
    {
        problem->costs = original_costs;
        perturbed = false;

        if (!checkPrimalFeasible() && checkDualFeasible()) 
            callDualSolver(method);

        else if (checkPrimalFeasible() && !checkDualFeasible())
            callPrimalSolver();

        else if (!checkPrimalFeasible() && !checkDualFeasible())
        {
            Phase1OutStatus status_code = minimizeDualInfeasibility();
            if (status_code == Phase1OutStatus::Solved)
                callDualSolver(method);
        }
        solution.x = x(0, problem->logicals_size);
        solution.Z = obj_func_val;
        return solution;
    }
    else if (!status_code && perturbed) // dual unbound case
    {
        problem->costs = original_costs;
        perturbed = false;

        Phase1OutStatus status_code = minimizeDualInfeasibility();
        if (status_code == Phase1OutStatus::Solved)
            callDualSolver(method);

        solution.x = x(0, problem->logicals_size);
        solution.Z = obj_func_val;
        return solution;
    }
    else  // without perturbation case
    {
        solution.x = x(0, problem->logicals_size);
        solution.Z = obj_func_val;
        return solution;
    }
    
    
    
}


//----------------------------------------------------------------------------------------
// Constructor for solver, additionally finds dual feasible basis
//----------------------------------------------------------------------------------------
sequentialDualSimplex::sequentialDualSimplex(
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
bool sequentialDualSimplex::checkPerturbNeed() const
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
ValuesVector sequentialDualSimplex::initBetaWeights()
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
void sequentialDualSimplex::perturbCosts()
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
}


//----------------------------------------------------------------------------------------
// Weight for perturbation depends on two goals "keep nonzero count low" and 
// "resolver degeneracy"
//----------------------------------------------------------------------------------------
double sequentialDualSimplex::getWeight(const size_t i) const
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
size_t sequentialDualSimplex::calcNonzeroInColumn(const size_t i) const
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
bool sequentialDualSimplex::checkPrimalFeasible() const
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
bool sequentialDualSimplex::checkDualFeasible() const
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
void sequentialDualSimplex::calcDualInfeasible()
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
size_t sequentialDualSimplex::counterDualInfeasible() const
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
// Phase 1 method for finding dual fesaible basis, based on article:
// E. Kostina. The long step rule in the bounded-variable dual simplex method:
// Numerical experiments. Mathematical Methods of Operations Research, 55:413–
// 429, 2002.
//----------------------------------------------------------------------------------------
sequentialDualSimplex::Phase1OutStatus sequentialDualSimplex::minimizeDualInfeasibility()
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
// Phase 1 method for finding dual fesaible basis, based on article:
// P. Q. Pan. The most-obtuse-angle row pivot rule for achieving dual feasibility:
// a computational study. European Journal of Operations Research, 101(1):167–
// 176, 1997.
//----------------------------------------------------------------------------------------
sequentialDualSimplex::Phase1OutStatus sequentialDualSimplex::panMathod()
{
    // (Step 1) Initialization
    ValuesVector y(problem->constraints_size);
    y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
    d.setValues(problem->costs(non_basis_indexes) - AN.T().dot(y), non_basis_indexes);
    Phase1OutStatus status;

    while (true)
    {
        // (Step 2) Select enetring variable
        IndexVector infeasibility_indexes;
        for (size_t i = 0; i < non_basis_size; i++)
        {
            size_t j = non_basis_indexes[i];
            if ((problem->bound_type[j] == BoundaryType::Upper || 
                problem->bound_type[j] == BoundaryType::Free) && d[j] > 0)
                infeasibility_indexes.push_back(i);
            else if ((problem->bound_type[j] == BoundaryType::Lower || 
                problem->bound_type[j] == BoundaryType::Free) && d[j] < 0)
                infeasibility_indexes.push_back(i);
        }

        if (checkDualFeasible())
        {
            calcDualInfeasible();
            status = Phase1OutStatus::Solved;
            break;
        }

        size_t q_idx = infeasibility_indexes[0];
        size_t q = non_basis_indexes[q_idx];
        double d_max = d[q];
        for (auto i : infeasibility_indexes)
        {
            size_t j = non_basis_indexes[i];
            if (fabs(d[j]) > fabs(d_max))
            {
                q_idx = i;
                q = j;
                d_max = d[j];
            }
        }
        
        // (Step 3) FTran
        ValuesVector alpha_q;
        alpha_q = linalg::PFIsolve(B_eta_repr, AN(q_idx), false);

       
        // (Step 4) Select leaving variable
        IndexVector check_indexes;
        for (size_t i = 0; i < problem->constraints_size; i++)
        {
            size_t j = basis_indexes[i];
            if (((problem->bound_type[j] == BoundaryType::Lower || problem->bound_type[j] == BoundaryType::Fixed) && alpha_q[i] < 0) ||
                ((problem->bound_type[j] == BoundaryType::Upper || problem->bound_type[j] == BoundaryType::Fixed) && alpha_q[i] > 0))
                check_indexes.push_back(i);
        }

        if (!check_indexes.size())
        {
            solution.solved = false;
            solution.message = "dual infeasible";
            status = Phase1OutStatus::DualInfeas;
            break;
        }
        
        size_t p_idx = check_indexes[0];
        size_t p = basis_indexes[p_idx];       
        double alpha_q_tmp = alpha_q[p_idx];
        for (auto i : check_indexes)
        {
            if (fabs(alpha_q[i]) >= fabs(alpha_q_tmp))
            {
                p = basis_indexes[i];
                p_idx = i;
                alpha_q_tmp = alpha_q[i];
            } 
        }

        // (Step 5) BTran
        ValuesVector rho(problem->constraints_size);
        rho = linalg::PFIsolve(B_eta_repr, linalg::unit(problem->constraints_size, p_idx), true);

        // (Step 6) Pivot row
        ValuesVector alpha(non_basis_size);
        alpha = AN.T().dot(rho);

        double  theta = d[q] / alpha[q_idx];

        // (Step 7) Basis change and update
        for (size_t i = 0; i < non_basis_indexes.size(); i++)
        {
            size_t j = non_basis_indexes[i];
            d[j] -= theta * alpha[i];       
        }
        d[p] = -theta;
        d[q] = 0;

        ValuesVector new_eta_matrix(problem->constraints_size);
        for (size_t i = 0; i < problem->constraints_size; i++)
            new_eta_matrix[i] = (i != p_idx) ? -alpha_q[i] / alpha_q[p_idx] : 1 / alpha_q[p_idx];

        basis_indexes[p_idx] =  q;
        non_basis_indexes[q_idx] = p;
      
        B.swapColumn(AN, p_idx, q_idx);
        B_eta_repr.push_back(EtaMatrix(new_eta_matrix, p_idx));

        // if (!linalg::checkPFIdecompose(B_eta_repr, B)) std::cerr << "Incorrect decompose" << std::endl;
    }
    return status;
}


//----------------------------------------------------------------------------------------
// Phase 2 simple method for finding solution, based on article:
// Dipl. Inform. Achim Koberstein. The Dual Simplex Method, Techniques for a fast and 
// stable implementation: for a fast and stable implementation, November 2005
//----------------------------------------------------------------------------------------
bool sequentialDualSimplex::elaboratedMethod()
{
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Phase 2 : started" << std::endl;

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
        double delta;
        size_t p;
        size_t p_idx;

        if (checkPrimalFeasible())
        {
            solution.solved = true;
            solution.message = "optimal solution";
            break;
        }
        
        bool is_lower = false;
        double max_weight = 0;
        for (size_t i = 0; i < problem->constraints_size; i++)
        {
            size_t j = basis_indexes[i];
            double delta_tmp = 0;
            bool is_lower_tmp; 
           
            if ((problem->bound_type[j] == BoundaryType::Boxed || 
                problem->bound_type[j] == BoundaryType::Lower ||
                problem->bound_type[j] == BoundaryType::Fixed ) && 
                problem->lower_bound[j] - x[j] > EPS_BOUND)
            {
                is_lower_tmp = true;
                delta_tmp = x[j] - problem->lower_bound[j];
                
            } 
            else if ((problem->bound_type[j] == BoundaryType::Boxed || 
                      problem->bound_type[j] == BoundaryType::Upper ||
                      problem->bound_type[j] == BoundaryType::Fixed ) && 
                      x[j] - problem->upper_bound[j] > EPS_BOUND)
            {
                is_lower_tmp = false;
                delta_tmp = x[j] - problem->upper_bound[j];
            }

            double weight_tmp = pow(delta_tmp, 2) / beta[i];
            if (weight_tmp > max_weight && delta_tmp != 0)
            {
                p = j;
                p_idx = i;
                delta  = delta_tmp;
                max_weight = weight_tmp;
                is_lower = is_lower_tmp;
            }
            
        }
        
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
            solution.solved = false;
            solution.message = "dual unbounded";
            break;
        }

        // (Step 6) FTran
        ValuesVector alpha_q;
        alpha_q = linalg::PFIsolve(B_eta_repr, AN(q_idx), false);

        // (Step 7) Basis change and update

        // Update d accoprding to BRFT
        ValuesVector column_change(problem->constraints_size);
        IndexVector infeas_idx;
        double delta_z = 0;

        for (size_t i = 0; i < non_basis_indexes.size(); i++)
        {
            size_t j = non_basis_indexes[i];
            d[j] = d[j] - theta * alpha_p[i];  
            if (problem->bound_type[j] == BoundaryType::Boxed)
            {
            
                if (x[j] == problem->lower_bound[j] && d[j] < 0)
                {
                    infeas_idx.push_back(j);
                    column_change += AN(i) * (problem->upper_bound[j] - problem->lower_bound[j]);
                    delta_z += problem->costs[j] * (problem->upper_bound[j] - problem->lower_bound[j]);
                }  
                else if (x[j] == problem->upper_bound[j] && d[j] > 0)
                {
                    infeas_idx.push_back(j);
                    column_change -= AN(i) * (problem->upper_bound[j] - problem->lower_bound[j]);
                    delta_z -= problem->costs[j] * (problem->upper_bound[j] - problem->lower_bound[j]);
                }   
            }          
        }
        d[p] = -theta;
        d[q] = 0;

        ValuesVector delta_xB;
        if (!infeas_idx.size())
        {
            delta_xB = linalg::PFIsolve(B_eta_repr, column_change, false);
            x(basis_indexes) -= delta_xB;
            for (size_t i = 0; i < problem->constraints_size; i++)
            {
                size_t j = basis_indexes[i];
                delta_z -= problem->costs[j] * delta_xB[i];
            }    
        }

        obj_func_val += delta_z;

        // Update B and xB, DSE weights
        double theta_P = delta / alpha_q[p_idx];
        ValuesVector new_eta_matrix(problem->constraints_size);
        ValuesVector tau(problem->constraints_size);
        tau = linalg::PFIsolve(B_eta_repr, rho, false);
        for (size_t i = 0; i < problem->constraints_size; i++)
        {
            size_t j = basis_indexes[i];
            x[j] = x[j] - theta_P * alpha_q[i];  
            new_eta_matrix[i] = (i != p_idx) ? -alpha_q[i] / alpha_q[p_idx] : 1 / alpha_q[p_idx]; 
            beta[i] = (i != p_idx) ? beta[i] - 2 * alpha_q[i] / alpha_q[p_idx] * tau[i]  + pow(alpha_q[i] / alpha_q[p_idx], 2) * beta[p_idx] : beta[i]; 
        }  
        beta[p_idx] = beta[p_idx] / pow(alpha_q[p_idx], 2);
        x[q] = x[q] + theta_P;
       
       // Update basis
        basis_indexes[p_idx] =  q;
        non_basis_indexes[q_idx] = p;
        B.swapColumn(AN, p_idx, q_idx);
        B_eta_repr.push_back(EtaMatrix(new_eta_matrix, p_idx));

        // Flip bounds
        for (auto j : infeas_idx)
        {
            if (x[j] == problem->lower_bound[j])
                x[j] = problem->upper_bound[j];
            else 
                x[j] = problem->lower_bound[j]; 
        }
        obj_func_val += theta * delta;

        iteration += 1;
        if (fabs(theta) < EPS_A) cycle_num += 1;
    }
    std::cout << "Phase 2 : ended" << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time spent: " << duration_ns.count() << " ms\n";
    return solution.solved;
}


//----------------------------------------------------------------------------------------
// Phase 2 simple method for finding solution, based on article:
// Dipl. Inform. Achim Koberstein. The Dual Simplex Method, Techniques for a fast and 
// stable implementation: for a fast and stable implementation, November 2005
//----------------------------------------------------------------------------------------
bool sequentialDualSimplex::simpleRatioMethod()
{
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Phase 2 : started" << std::endl;
    // (Step 1) Initialization
    ValuesVector y(problem->constraints_size);

    x.setValues(linalg::PFIsolve(B_eta_repr, problem->RHS - AN.dot(x(non_basis_indexes)), false), basis_indexes);
    y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
    d.setValues(problem->costs(non_basis_indexes) - AN.T().dot(y), non_basis_indexes);
    obj_func_val = problem->costs.dot(x);

    size_t iteration = 0;
    while (true)
    {
        #ifdef DEBUG
        std::cout << iteration << " : Z = "<< obj_func_val << std::endl;
        #endif
        // (Step 2) Pricing
        double delta;
        size_t p;
        size_t p_idx;
        bool is_lower = false;

        if (checkPrimalFeasible())
        {
            solution.solved = true;
            solution.message = "optimal solution";
            break;
        }
            

        for (size_t i = 0; i < problem->constraints_size; i++)
        {
            p = basis_indexes[i];
            p_idx = i;
            if ((problem->bound_type[p] == BoundaryType::Boxed || 
                problem->bound_type[p] == BoundaryType::Lower ||
                problem->bound_type[p] == BoundaryType::Fixed ) && 
                problem->lower_bound[p] - x[p] > EPS_BOUND)
            {
                is_lower = true;
                delta = x[p] - problem->lower_bound[p];
                break;
            } 
            else if ((problem->bound_type[p] == BoundaryType::Boxed || 
                      problem->bound_type[p] == BoundaryType::Upper ||
                      problem->bound_type[p] == BoundaryType::Fixed ) && 
                      x[p] - problem->upper_bound[p] > EPS_BOUND)
            {
                delta = x[p] - problem->upper_bound[p];
                break;
            }
        }
        
        // (Step 3) BTran
        ValuesVector rho(problem->constraints_size);
        rho = linalg::PFIsolve(B_eta_repr, linalg::unit(problem->constraints_size, p_idx), true);
       
        // (Step 4) Pivot row
        ValuesVector alpha_p(non_basis_size);
        alpha_p = AN.T().dot(rho);

        // (Step 5) Ratio Test
        ValuesVector tmp_alpha_p(non_basis_size);
        if (is_lower)
            tmp_alpha_p = -alpha_p;
        else
            tmp_alpha_p = alpha_p;

        IndexVector F;
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

        if (F.size() == 0)
        {
            solution.solved = false;
            solution.message = "dual unbounded";
            break;
        }
        
        size_t q = non_basis_indexes[F[0]];
        size_t q_idx = F[0];
        double theta = (d[q] / tmp_alpha_p[q_idx]);
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
        theta = d[q] / alpha_p[q_idx];

        // (Step 6) FTran
        ValuesVector alpha_q;
        alpha_q = linalg::PFIsolve(B_eta_repr, AN(q_idx), false);


        // (Step 7) Basis change and update
        obj_func_val += theta * delta;
        for (size_t i = 0; i < non_basis_indexes.size(); i++)
        {
            size_t j = non_basis_indexes[i];
            d[j] = d[j] - theta * alpha_p[i];    
        }
        d[p] = -theta;
        d[q] = 0;

        double theta_P = delta / alpha_q[p_idx];
        ValuesVector new_eta_matrix(problem->constraints_size);
        for (size_t i = 0; i < problem->constraints_size; i++)
        {
            size_t j = basis_indexes[i];
            x[j] = x[j] - theta_P * alpha_q[i];  
            new_eta_matrix[i] = (i != p_idx) ? -alpha_q[i] / alpha_q[p_idx] : 1 / alpha_q[p_idx]; 
        }  
        x[q] = x[q] + theta_P;

        basis_indexes[p_idx] =  q;
        non_basis_indexes[q_idx] = p;

        B.swapColumn(AN, p_idx, q_idx);
        B_eta_repr.push_back(EtaMatrix(new_eta_matrix, p_idx));

        iteration += 1;
    }
    std::cout << "Phase 2 : ended" << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time spent: " << duration_ns.count() << " ms\n";
    return solution.solved;
}
