#include "cudaDualSimplex.hpp"


//----------------------------------------------------------------------------------------
// Init params, setting data according to index arrays
//----------------------------------------------------------------------------------------
void CudaDualSimplex::initDualSimplex()
{
    x = ValuesVector(problem->problem_size);
    d = ValuesVector(problem->problem_size);   

    AN = problem->A(non_basis_indexes);
    B = problem->A(basis_indexes);

    device_AN = CudaSparseMatrix(AN);
    device_B = CudaSparseMatrix(B);

    device_xB = CudaDenseVector(x(basis_indexes));
    device_xN = CudaDenseVector(x(non_basis_indexes));

    device_dB = CudaDenseVector(d(basis_indexes));
    device_dN = CudaDenseVector(d(non_basis_indexes));

    cusolverSpCreate(&sl_handle);
    cusolverRfCreate(&rf_handle);
    cusparseCreate(&sp_handle);
    cublasCreate(&cu_handle);

    device_B.LUdecompose(sl_handle, B_repr);
    B_repr.setup(rf_handle, device_B);
}

//----------------------------------------------------------------------------------------
// Convert string to exiting methods for presolver
//----------------------------------------------------------------------------------------
PresolverMethods CudaDualSimplex::stringToPreSolverMethod(
    const std::string& method_name
)
{
    static const std::unordered_map<std::string, PresolverMethods> methodMap = {
        {"minInfeas", PresolverMethods::minDualInfeasibility},
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
SolverMethods CudaDualSimplex::stringToSolverMethod(
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
// Choose presolver 
//----------------------------------------------------------------------------------------
CudaDualSimplex::Phase1OutStatus CudaDualSimplex::callPresolver(const PresolverMethods method)
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
bool CudaDualSimplex::callDualSolver(const SolverMethods method)
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
bool CudaDualSimplex::callPrimalSolver()
{
   return true;
}


//----------------------------------------------------------------------------------------
// Initialize weights for DSE according to
// John J. Forrest and Donald Goldfarb. Steepest-edge simplex algorithms for
// linear programming. Math. Program., 57(3):341–374, 1992.
//----------------------------------------------------------------------------------------
void CudaDualSimplex::initBetaWeights()
{
    ValuesVector beta(problem->constraints_size);
    for (size_t i = 0; i < problem->constraints_size; i++)
    {
        CudaDenseVector sol;
        B_repr.solve(rf_handle, linalg::unit(problem->constraints_size, i), sol , true);
        beta[i] = sol.norm(cu_handle);
    }
    device_beta = CudaDenseVector()
}


//----------------------------------------------------------------------------------------
// Phase 1 method for finding dual fesaible basis, based on article:
// E. Kostina. The long step rule in the bounded-variable dual simplex method:
// Numerical experiments. Mathematical Methods of Operations Research, 55:413–
// 429, 2002.
//----------------------------------------------------------------------------------------
CudaDualSimplex::Phase1OutStatus CudaDualSimplex::minimizeDualInfeasibility()
{
    // (Step 1) Initialization
    Phase1OutStatus status;
    ValuesVector y(problem->constraints_size);
    y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
    d.setValues(problem->costs(non_basis_indexes) - AN.dot(y, true), non_basis_indexes);
    
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
            d.setValues(problem->costs(non_basis_indexes) - AN.dot(y, true), non_basis_indexes);
        }

        // if (cycle_num > RESTART_CYCLE)
        // {
        //     status = Phase1OutStatus::NeedRestart;
        //     break;
        // }
        
        
        // (Step 2) Pricing
        size_t p, p_idx;
        double max_weight = 0;
        bool no_candidates =true;
        for (size_t i = 0; i < problem->constraints_size; i++)
        {
            size_t j = basis_indexes[i];
            if (((problem->bound_type[j] != BoundaryType::Free && problem->bound_type[j] != BoundaryType::Upper) && f[i] > EPS_BOUND) ||
                ((problem->bound_type[j] != BoundaryType::Free && problem->bound_type[j] != BoundaryType::Lower) && f[i] < -EPS_BOUND))
            {
                double weight_tmp = pow(f[i], 2) / beta[i];
                if (weight_tmp > max_weight && blocked_p.find(j) == blocked_p.end())
                {
                    no_candidates = false;
                    p = j;
                    p_idx = i;
                    max_weight = weight_tmp;
                }
            }
            
        }

        if (checkDualFeasible() || no_candidates)
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
        alpha = AN.dot(rho, true);

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
        if (fabs(theta) < EPS_A) 
            cycle_num += 1;
        else
            cycle_num = 0;

        #ifdef DEBUG
        std::cout << iteration << " : Z = "<< obj_func_val << " inf_num = " << counterDualInfeasible() << " p = " << p << " q = " << q << std::endl;  
        #endif
    }
    return status;
}


//----------------------------------------------------------------------------------------
// Phase 2 simple method for finding solution, based on article:
// Dipl. Inform. Achim Koberstein. The Dual Simplex Method, Techniques for a fast and 
// stable implementation: for a fast and stable implementation, November 2005
//----------------------------------------------------------------------------------------
bool CudaDualSimplex::elaboratedMethod()
{
    
    // (Step 1) Initialization
    ValuesVector y(problem->constraints_size);
    ValuesVector beta(problem->constraints_size);

    problem->RHS = problem->RHS - AN.dot(x(non_basis_indexes), false);
    x.setValues(linalg::PFIsolve(B_eta_repr, problem->RHS, false), basis_indexes);
    y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
    d.setValues(problem->costs(non_basis_indexes) - AN.dot(y, true), non_basis_indexes);
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
            d.setValues(problem->costs(non_basis_indexes) - AN.dot(y, true), non_basis_indexes);
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
        alpha_p = AN.dot(rho, true);

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
            x.setValues(x(basis_indexes) - delta_xB, basis_indexes);
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
    std::cout << "iterations = " << iteration << std::endl;
    return solution.solved;
}

