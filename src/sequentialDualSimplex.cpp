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
void sequentialDualSimplex::presolve(const std::string& presolver_method_name)
{
    auto method = stringToPreSolverMethod(presolver_method_name);
    if (checkPerturbNeed())
        perturbCosts();
    
    callPresolver(method); 
}


//----------------------------------------------------------------------------------------
// Choose presolver 
//----------------------------------------------------------------------------------------
bool sequentialDualSimplex::callPresolver(const sequentialDualSimplex::PresolverMethods method)
{
    bool status;
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
            bool status_code = panMathod();
            if (status_code)
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

        bool status_code = panMathod();
        if (status_code)
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
    presolve(presolver_method_name);
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
            if (fabs(d[i]) < EPS_D)
                return false;
            break;

        case BoundaryType::Upper:
            if (d[i] > EPS_D)
                return false;
            break;

        case BoundaryType::Lower:
            if (d[i] < -EPS_D)
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
// Phase 1 method for finding dual fesaible basis, based on article:
// E. Kostina. The long step rule in the bounded-variable dual simplex method:
// Numerical experiments. Mathematical Methods of Operations Research, 55:413–
// 429, 2002.
//----------------------------------------------------------------------------------------
bool sequentialDualSimplex::minimizeDualInfeasibility()
{
    // (Step 1) Initialization
    bool status;
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
            problem->bound_type[j] == BoundaryType::Free) && d[j] > 0)
            inf_u_indexes.push_back(j);
        else if ((problem->bound_type[j] == BoundaryType::Lower || 
            problem->bound_type[j] == BoundaryType::Free) && d[j] < 0)
            inf_l_indexes.push_back(j);

        if (problem->bound_type[j] == BoundaryType::Free && d[j] < 0)
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
    
    size_t iteration = 0;
    size_t cycle_num = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 1);


    while (true)
    {
        if (!perturbed && cycle_num > MAX_CYCLE) 
        {
            perturbCosts();
            y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
            d.setValues(problem->costs(non_basis_indexes) - AN.T().dot(y), non_basis_indexes);
        }
        
        // (Step 2) Pricing
        IndexVector check_indexes;
        for (size_t i = 0; i < problem->constraints_size; i++)
        {
            size_t j = basis_indexes[i];
            if (((problem->bound_type[j] != BoundaryType::Free && problem->bound_type[j] != BoundaryType::Upper) && f[i] > 0) ||
                ((problem->bound_type[j] != BoundaryType::Free && problem->bound_type[j] != BoundaryType::Lower) && f[i] < 0))
                check_indexes.push_back(i);
        }

        if (checkDualFeasible())
        {
            calcDualInfeasible();
            if (obj_func_val == 0)
            {
                status = true;
            }
            else if (obj_func_val > 0)
            {
                solution.solved = false;
                solution.message = "dual infeasible";
                status = false;
            }
            break;
        }

        size_t p, p_idx;
        size_t q, q_idx;

        double opt_step = 0;

        ValuesVector alpha(non_basis_size);
        ValuesVector rho(problem->constraints_size);
        double theta;

        bool start = true;

        for (auto p_idx_tmp : check_indexes)
        {
            size_t p_tmp = basis_indexes[p_idx_tmp]; 
            
            // (Step 3) BTran
            rho = linalg::PFIsolve(B_eta_repr, linalg::unit(problem->constraints_size, p_idx_tmp), true);
            
            // (Step 4) Pivot row
            alpha = AN.T().dot(rho);

            // (Step 5) Ratio Test
            ValuesVector alpha_tmp(non_basis_size);
            ValuesVector f_tmp(problem->constraints_size);
            if (f[p_idx_tmp] > 0)
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

            if (!F.size()) continue;

            size_t q_idx_tmp = F[0];
            size_t q_tmp = non_basis_indexes[q_idx_tmp];;
            theta = d[q_tmp] / alpha_tmp[q_idx_tmp];
            for (auto i : F)
            {   
                size_t j = non_basis_indexes[i];
                double theta_tmp = d[j] / alpha_tmp[i];
                if (theta_tmp < theta || (fabs(theta_tmp - theta) < EPS_BOUND && distrib(gen) == 1)) 
                {
                    theta = theta_tmp;
                    q_tmp = j;
                    q_idx_tmp = i;
                } 
            }

            theta = d[q_tmp] / alpha[q_idx_tmp];

            if (f[p_idx_tmp] * theta < opt_step || (fabs(fabs(opt_step) - fabs(f[p_idx_tmp])) < EPS_BOUND && distrib(gen) == 1) || start) 
            {
                p = p_tmp;
                p_idx = p_idx_tmp;
                q = q_tmp;
                q_idx = q_idx_tmp;
                opt_step = f[p_idx_tmp] * theta;
                start = false;
            } 
        }

        rho = linalg::PFIsolve(B_eta_repr, linalg::unit(problem->constraints_size, p_idx), true);
        alpha = AN.T().dot(rho);

        theta = d[q] / alpha[q_idx];


        // ValuesVector alpha_tmp(non_basis_size);
        // ValuesVector f_tmp(problem->constraints_size);
        // if (f[p_idx] > 0 && obj_func_val < 0)
        // {
        //     alpha_tmp = -alpha;
        //     f_tmp = -f;
        // }
        // else
        // {
        //     alpha_tmp = alpha;
        //     f_tmp = f;
        // }
        
        // IndexVector F, F_reserved;
        // for (size_t i = 0; i < non_basis_indexes.size(); i++)
        // {
        //     size_t j = non_basis_indexes[i];
        //     if (((d[j] >= 0 && alpha_tmp[i] > 0) ||
        //         (d[j] <= 0 && alpha_tmp[i] < 0)))
        //         F.push_back(i);
        // }
        
        // size_t q_idx = F[0];
        // size_t q = non_basis_indexes[q_idx];;
        // double theta = 0;
        // ValuesVector columns_change_tmp(problem->constraints_size);
        // while (!F.size() && f_tmp[p_idx] <= 0)
        // {
        //     q_idx = F[0];
        //     q = non_basis_indexes[q_idx];;
        //     theta = d[q] / alpha_tmp[q_idx];
        //     for (auto i : F)
        //     {   
        //         size_t j = non_basis_indexes[i];
        //         double theta_tmp = d[j] / alpha_tmp[i];
        //         if (theta_tmp < theta || (fabs(theta_tmp - theta) < EPS_BOUND && distrib(gen) == 1)) 
        //         {
        //             theta = theta_tmp;
        //             q = j;
        //             q_idx = i;
        //         } 
        //     }

        //     auto _ = std::remove(F.begin(), F.end(), q);

        //     if (std::count(inf_f_indexes.begin(), inf_f_indexes.end(), q) && f_tmp[p_idx] <= 0)
        //         f_tmp[p_idx] += 2 * fabs(alpha[q_idx]);
        //     else
        //         f_tmp[p_idx] += fabs(alpha[q_idx]);

        //     if (F.size() != 0 && f[p_idx] <= 0)
        //     {
        //         if (std::count(inf_f_indexes.begin(), inf_f_indexes.end(), q))
        //             columns_change_tmp += AN(q_idx) * 2;
        //         else
        //             columns_change_tmp += AN(q_idx);
        //     }
        //     else if (d[q] < 0)
        //     {
        //         if (std::count(inf_f_indexes.begin(), inf_f_indexes.end(), q))
        //             columns_change_tmp -= AN(q_idx) * 2;
        //         else
        //             columns_change_tmp -= AN(q_idx);
        //     }
        //     else
        //     {
        //         if (alpha[q_idx] > 0)
        //             columns_change_tmp += AN(q_idx);
        //         else
        //             columns_change_tmp -= AN(q_idx);
        //     }
        // }

        // theta = d[q] / alpha[q_idx];
        

        // (Step 6) FTran
        ValuesVector alpha_q;
        alpha_q = linalg::PFIsolve(B_eta_repr, AN(q_idx), false);

        // (Step 7) Basis change and update
        ValuesVector tau(problem->constraints_size);
        tau = linalg::PFIsolve(B_eta_repr, rho, false);

        obj_func_val = obj_func_val - theta * f[p_idx];
        for (size_t i = 0; i < non_basis_indexes.size(); i++)
        {
            size_t j = non_basis_indexes[i];
            d[j] -= theta * alpha[i];    
        }
        d[p] = -theta;
        d[q] = 0;

        // ValuesVector delta_f;
        // delta_f = linalg::PFIsolve(B_eta_repr, columns_change_tmp, false);
        // f += delta_f;

        for (size_t i = 0; i < problem->constraints_size; i++)
            f[i] = (i != p_idx) ? f[i] - alpha_q[i] / alpha_q[p_idx] * f[p_idx] : f[p_idx];
        f[p_idx] = f[p_idx] / alpha_q[p_idx];
        
        ValuesVector new_eta_matrix(problem->constraints_size);
        for (size_t i = 0; i < problem->constraints_size; i++)
            new_eta_matrix[i] = (i != p_idx) ? -alpha_q[i] / alpha_q[p_idx] : 1 / alpha_q[p_idx];

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
        iteration += 1;
        if (fabs(theta) < EPS_A) cycle_num += 1;

        std::cout << iteration << " : Z = "<< obj_func_val << " p = " << p << " q = " << q << "opt_step = " << opt_step << std::endl;

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
bool sequentialDualSimplex::panMathod()
{
    // (Step 1) Initialization
    ValuesVector y(problem->constraints_size);
    y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
    d.setValues(problem->costs(non_basis_indexes) - AN.T().dot(y), non_basis_indexes);
    bool status;

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
            status = false;
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
            status = false;
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
    // (Step 1) Initialization
    ValuesVector y(problem->constraints_size);

    problem->RHS = problem->RHS - AN.dot(x(non_basis_indexes));
    x.setValues(linalg::PFIsolve(B_eta_repr, problem->RHS, false), basis_indexes);
    y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
    d.setValues(problem->costs(non_basis_indexes) - AN.T().dot(y), non_basis_indexes);
    obj_func_val = problem->costs.dot(x);

    while (true)
    {
        
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
            

        for (size_t i = 0; i < problem->constraints_size; i++)
        {
            p = basis_indexes[i];
            p_idx = i;
            if ((problem->bound_type[p] == BoundaryType::Boxed || 
                problem->bound_type[p] == BoundaryType::Lower) && 
                x[p] < problem->lower_bound[p])
            {
                delta = x[p] - problem->lower_bound[p];
                break;
            } 
            else if ((problem->bound_type[p] == BoundaryType::Boxed || 
                      problem->bound_type[p] == BoundaryType::Upper) && 
                      x[p] > problem->upper_bound[p])
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
        if (x[p] < problem->lower_bound[p])
            tmp_alpha_p = -alpha_p;
        else
            tmp_alpha_p = alpha_p;

        IndexVector F;
        for (size_t i = 0; i < non_basis_size; i++)
        {
            size_t j = non_basis_indexes[i];
            if ((tmp_alpha_p[i] > 0 && x[j] == problem->lower_bound[j] &&
                (problem->bound_type[j] == BoundaryType::Lower || problem->bound_type[j] == BoundaryType::Boxed)) ||
                (tmp_alpha_p[i] < 0 && x[j] == problem->upper_bound[j] && 
                (problem->bound_type[j] == BoundaryType::Upper || problem->bound_type[j] == BoundaryType::Boxed)) ||
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
            if (theta_tmp < theta)
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
        // d[p] = -theta;
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

        // obj_func_val = problem->costs.dot(x);

        basis_indexes[p_idx] =  q;
        non_basis_indexes[q_idx] = p;
        B.swapColumn(AN, p_idx, q_idx);
        B_eta_repr.push_back(EtaMatrix(new_eta_matrix, p_idx));
    }
    return solution.solved;
}


//----------------------------------------------------------------------------------------
// Phase 2 simple method for finding solution, based on article:
// Dipl. Inform. Achim Koberstein. The Dual Simplex Method, Techniques for a fast and 
// stable implementation: for a fast and stable implementation, November 2005
//----------------------------------------------------------------------------------------
bool sequentialDualSimplex::simpleRatioMethod()
{
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
        std::cout << iteration << " : Z = "<< obj_func_val << std::endl;
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

        // obj_func_val = problem->costs.dot(x);

        basis_indexes[p_idx] =  q;
        non_basis_indexes[q_idx] = p;

        B.swapColumn(AN, p_idx, q_idx);
        B_eta_repr.push_back(EtaMatrix(new_eta_matrix, p_idx));

        iteration += 1;
        // if (!linalg::checkPFIdecompose(B_eta_repr, B)) std::cerr << "Incorrect decompose" << std::endl;
    }
    std::cout << "Phase 2 : ended" << std::endl;
    return solution.solved;
}
