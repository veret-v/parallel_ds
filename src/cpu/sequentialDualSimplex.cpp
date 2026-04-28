#include "sequentialDualSimplex.hpp"


//----------------------------------------------------------------------------------------
// Init params, setting data according to index arrays
//----------------------------------------------------------------------------------------
void SequentialDualSimplex::initDualSimplex()
{
    std::cout << "Solver initialization : basis columns selected" << std::endl;

    x = ValuesVector(full_size);
    d = ValuesVector(full_size);    
    B.resetData(problem->A, basis_indexes);
    B_eta_repr.clear();
    B.LUdecompose();

    std::cout << "Solver initialization : attributes setted" << std::endl; 
}


void SequentialDualSimplex::solveLinSys(
    ValuesVector&& rhs, 
    ValuesVector& sol,
    bool transpose
)
{
    if (transpose)
    {
        linalg::PFIsolve(B_eta_repr, rhs, sol, transpose);
        ValuesVector new_rhs = sol;
        B.solve(new_rhs, sol, transpose);
    }
    else
    {
        B.solve(rhs, sol, transpose);
        ValuesVector new_rhs = sol;
        linalg::PFIsolve(B_eta_repr, new_rhs, sol, transpose);
    }
}


void SequentialDualSimplex::solveLinSys(
    ValuesVector& rhs, 
    ValuesVector& sol,
    bool transpose
)
{
    if (transpose)
    {
        linalg::PFIsolve(B_eta_repr, rhs, sol, transpose);
        ValuesVector new_rhs = sol;
        B.solve(new_rhs, sol, transpose);
    }
    else
    {
        B.solve(rhs, sol, transpose);
        ValuesVector new_rhs = sol;
        linalg::PFIsolve(B_eta_repr, new_rhs, sol, transpose);
    }
    
}


//----------------------------------------------------------------------------------------
// Convert string to exiting methods for presolver
//----------------------------------------------------------------------------------------
PresolverMethods SequentialDualSimplex::stringToPreSolverMethod(
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
SolverMethods SequentialDualSimplex::stringToSolverMethod(
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
Phase1OutStatus SequentialDualSimplex::callPresolver(const PresolverMethods method)
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
bool SequentialDualSimplex::callDualSolver(const SolverMethods method)
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
bool SequentialDualSimplex::callPrimalSolver()
{
   return true;
}


//----------------------------------------------------------------------------------------
// Initialize weights for DSE according to
// John J. Forrest and Donald Goldfarb. Steepest-edge simplex algorithms for
// linear programming. Math. Program., 57(3):341–374, 1992.
//----------------------------------------------------------------------------------------
ValuesVector SequentialDualSimplex::initBetaWeights()
{
    ValuesVector beta(basis_size);
    ValuesVector buff(basis_size);
    for (int i = 0; i < basis_size; i++)
    {
        solveLinSys(linalg::unit(problem->constraints_size, i), buff, true);
        beta[i] = buff.norm();
    }
    return beta;
}


//----------------------------------------------------------------------------------------
// Init reduced costs(d)
//----------------------------------------------------------------------------------------
void SequentialDualSimplex::initReducedCosts(ValuesVector& vec)
{
    solveLinSys(problem->costs(basis_indexes), vec, true);
    problem->A.dotUpdate(
        vec, problem->costs, 
        d, -1, 1, 
        non_basis_indexes, 
        SpmvOptions::UPDATE_T,
        true
    );
}


//----------------------------------------------------------------------------------------
// Phase 1 method for finding dual fesaible basis, based on article:
// E. Kostina. The long step rule in the bounded-variable dual simplex method:
// Numerical experiments. Mathematical Methods of Operations Research, 55:413–
// 429, 2002.
//----------------------------------------------------------------------------------------
Phase1OutStatus SequentialDualSimplex::minimizeDualInfeasibility()
{
    // (Step 1) Initialization
    Phase1OutStatus status;

    ValuesVector y(basis_size);                     // basis_size
    ValuesVector rho(basis_size);                   // basis_size
    ValuesVector f(basis_size);                     // basis_size
    ValuesVector columns_change(basis_size);        // basis_size
    ValuesVector beta(basis_size);                  // basis_size
    ValuesVector alpha(non_basis_size);             // non_basis_size
    ValuesVector alpha_q(basis_size);               // basis_size
    ValuesVector alpha_tmp(non_basis_size);         // non_basis_size
    ValuesVector f_tmp(basis_size);                 // basis_size
    ValuesVector new_eta_matrix(basis_size);        // basis_size
    ValuesVector tau(basis_size);                   // basis_size
    ValuesVector buff(basis_size);

    IndexVector inf_u_indexes;
    IndexVector inf_l_indexes;
    IndexVector stub_index;

    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> distrib(0, 1);
    std::uniform_int_distribution<> non_basis_rand(0, non_basis_size);
      

    solveLinSys(problem->costs(basis_indexes), y, true);
    problem->A.dotUpdate(
        y, problem->costs, 
        d, -1, 1, 
        non_basis_indexes, 
        SpmvOptions::UPDATE_T,
        true
    );

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

    obj_func_val = 0;
    for (auto i : inf_l_indexes)
    {
        obj_func_val += d[i];
        columns_change += problem->A(i);
    }

    for (auto i : inf_u_indexes)
    {
        obj_func_val -= d[i];
        columns_change -= problem->A(i);
    }

    solveLinSys(columns_change, f, false);
    
    beta = initBetaWeights();

    int iteration = 0;
    int cycle_num = 0;

    std::unordered_set<int> blocked_p;
    while (true)
    {
        iteration += 1;

        if (!perturbed && cycle_num > MAX_CYCLE) 
        {
            perturbCosts();
            initReducedCosts(y);
        }

        if (cycle_num > NEED_RESTART || 
            obj_func_val > INF || std::isnan(obj_func_val))
        {
            status = Phase1OutStatus::NeedRestart;
            break;
        }

        if (iteration % REFACT_FREQ == 0)
        {
            B_eta_repr.clear();
            B.resetData(problem->A, basis_indexes);
            B.LUdecompose();
        }

        // (Step 2) Pricing
        double max_weight = 0;
        double weight_tmp;
        bool candid_find = false;
        int p, p_idx;
        for (int i = 0; i < basis_size; i++)
        {
            int j = basis_indexes[i];
            if ((problem->bound_type[j] == BoundaryType::Lower && f[i] > EPS_BOUND) ||
                (problem->bound_type[j] == BoundaryType::Upper && f[i] < -EPS_BOUND) ||
                problem->bound_type[j] == BoundaryType::Boxed ||
                problem->bound_type[j] == BoundaryType::Fixed)
            {
                weight_tmp = pow(f[i], 2) / beta[i];
                candid_find = true;
                if (weight_tmp > max_weight && blocked_p.find(j) == blocked_p.end())
                {
                    p = j;
                    p_idx = i;
                    max_weight = weight_tmp;
                }
            }
        }

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

        // (Step 3) BTran
        solveLinSys(linalg::unit(basis_size, p_idx), rho, true);
        
        // (Step 4) Pivot row
        problem->A.dotUpdate(
            rho, problem->costs, 
            alpha, 1, 0, 
            non_basis_indexes, 
            SpmvOptions::UPDATE_T,
            false
        );

        // (Step 5) Ratio Test
        if (f[p_idx] > 0)
            alpha_tmp = -alpha;
        else
            alpha_tmp = alpha;

        IndexVector F;
        for (int i = 0; i < non_basis_indexes.size(); i++)
        {
            int j = non_basis_indexes[i];
            if (((d[j] >= 0 && alpha_tmp[i] > EPS_A) ||
                (d[j] <= 0 && alpha_tmp[i] < -EPS_A)))
                F.push_back(i);                
        }


        // rty to save
        if (!F.size()) 
        {
            blocked_p.insert(p);

            B_eta_repr.clear();
            B.resetData(problem->A, basis_indexes);
            B.LUdecompose();

            solveLinSys(problem->costs(basis_indexes), y, true);
            problem->A.dotUpdate(
                y, problem->costs, 
                d, -1, 1, 
                non_basis_indexes, 
                SpmvOptions::UPDATE_T,
                true
            );

            inf_u_indexes.clear();
            inf_l_indexes.clear();
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

            obj_func_val = 0;
            for (auto i : inf_l_indexes)
            {
                obj_func_val += d[i];
                columns_change += problem->A(i);
            }

            for (auto i : inf_u_indexes)
            {
                obj_func_val -= d[i];
                columns_change -= problem->A(i);
            }

            solveLinSys(columns_change, f, false);

            continue;
        }
        // end trying


        if (blocked_p.size())
            blocked_p.clear();

        int q_idx, q;
        double theta = INF;
        for (auto i : F)
        {   
            int j = non_basis_indexes[i];
            double theta_tmp = d[j] / alpha_tmp[i];
            if (theta_tmp < theta) 
            {
                theta = theta_tmp;
                q = j;
                q_idx = i;
            } 
            else if (fabs(theta_tmp - theta) < EPS_Z)
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
        buff = problem->A(q);
        solveLinSys(buff, alpha_q, false);

        // (Step 7) Basis change and update
        double infisib_corr = 0;
        if ((problem->bound_type[q] == BoundaryType::Upper || 
            problem->bound_type[q] == BoundaryType::Free) && d[q] > EPS_D)  
            infisib_corr = 1;
        else if ((problem->bound_type[q] == BoundaryType::Lower || 
            problem->bound_type[q] == BoundaryType::Free) && d[q] < -EPS_D) 
            infisib_corr = -1;
        
        
        obj_func_val = obj_func_val - theta * f[p_idx];
        for (int i = 0; i < non_basis_indexes.size(); i++)
        {
            int j = non_basis_indexes[i];
            d[j] -= theta * alpha[i];    
        }
        d[p] = -theta;
        d[q] = 0;

        solveLinSys(rho, tau, false);
        double theta_P = f[p_idx] / alpha_q[p_idx];
        for (int i = 0; i < basis_size; i++)
        {
            int j = basis_indexes[i];
            f[i] = (i != p_idx) ? f[i] - alpha_q[i] * theta_P : f[p_idx];
            new_eta_matrix[i] = (i != p_idx) ? -alpha_q[i] / alpha_q[p_idx] : 1 / alpha_q[p_idx]; 
            beta[i] = (i != p_idx) ? beta[i] - 2 * alpha_q[i] / alpha_q[p_idx] * tau[i]  + pow(alpha_q[i] / alpha_q[p_idx], 2) * beta[p_idx] : beta[i]; 
        }  
        beta[p_idx] = beta[p_idx] / pow(alpha_q[p_idx], 2);
        f[p_idx] = f[p_idx] / alpha_q[p_idx] + infisib_corr;

        basis_indexes[p_idx] =  q;
        non_basis_indexes[q_idx] = p;
    
        B_eta_repr.push_back(EtaMatrix(new_eta_matrix, p_idx));

        cycle_num = (fabs(theta) < EPS_A) ? cycle_num + 1 : 0;

        #ifdef DEBUG
            std::cout << iteration << " : Z = "<< obj_func_val << " inf_num = " << counterDualInfeasible() << " p_info:" << p << " q_info:" << q << " f:" << f[p_idx] << " beta:" << beta[p_idx] << " alpha_q:" << alpha_q[p_idx] << std::endl;  
        #endif
    }
    return status;
}


//----------------------------------------------------------------------------------------
// Phase 2 simple method for finding solution, based on article:
// Dipl. Inform. Achim Koberstein. The Dual Simplex Method, Techniques for a fast and 
// stable implementation: for a fast and stable implementation, November 2005
//----------------------------------------------------------------------------------------
bool SequentialDualSimplex::elaboratedMethod()
{
    // (Step 1) Initialization
    ValuesVector y(basis_size);
    ValuesVector beta(basis_size);
    ValuesVector rho(basis_size);
    ValuesVector alpha_p(non_basis_size);
    ValuesVector tmp_alpha_p(non_basis_size);
    ValuesVector alpha_q(basis_size);
    ValuesVector column_change(basis_size);
    ValuesVector delta_xB(basis_size);
    ValuesVector tau(basis_size);
    ValuesVector new_eta_matrix(basis_size);
    ValuesVector buff_sol(basis_size);

    IndexVector stub_index;
 
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> distrib(0, 1);
    std::uniform_int_distribution<> non_basis_rand(0, non_basis_size);
    
    problem->A.dotUpdate(
        x, problem->RHS, 
        problem->RHS, -1, 1, 
        non_basis_indexes, 
        SpmvOptions::UPDATE,
        true
    );

    solveLinSys(problem->RHS, buff_sol, false);
    x.setValues(buff_sol, basis_indexes);

    initReducedCosts(y);
    obj_func_val = problem->costs.dot(x);
    beta = initBetaWeights();

    int iteration = 0;
    int cycle_num = 0;

    while (true)
    {
        iteration += 1;
        if (iteration % REFACT_FREQ == 0)
        {
            B_eta_repr.clear();
            B.resetData(problem->A, basis_indexes);
            B.LUdecompose();
        }

        if (!perturbed && cycle_num > MAX_CYCLE) 
        {
            perturbCosts();
            initReducedCosts(y);
            obj_func_val = problem->costs.dot(x);
        }
        
        // (Step 2) Pricing
        double delta;
        int p;
        int p_idx;

        if (checkPrimalFeasible())
        {
            problem->solution.solved = true;
            problem->solution.message = "optimal solution";
            break;
        }

        bool is_lower = false;
        double max_weight = 0;
        for (int i = 0; i < basis_size; i++)
        {
            int j = basis_indexes[i];
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
        solveLinSys(linalg::unit(basis_size, p_idx), rho, true);

        // (Step 4) Pivot row
        problem->A.dotUpdate(
            rho, problem->costs, 
            alpha_p, 1, 0, 
            non_basis_indexes, 
            SpmvOptions::UPDATE_T,
            false
        );

        // (Step 5) Ratio Test
        ValuesVector tmp_alpha_p(non_basis_size);
        delta = fabs(delta);
        int sgn = (is_lower) ? -1 : 1;
        tmp_alpha_p = (is_lower) ? -alpha_p : alpha_p;

        IndexVector F;
        for (int i = 0; i < non_basis_size; i++)
        {
            int j = non_basis_indexes[i];
            if ((tmp_alpha_p[i] > EPS_A && fabs(x[j] - problem->lower_bound[j]) < EPS_Z  &&
                (problem->bound_type[j] == BoundaryType::Lower || 
                problem->bound_type[j] == BoundaryType::Boxed)) ||
                (tmp_alpha_p[i] < -EPS_A && fabs(x[j] - problem->upper_bound[j]) < EPS_Z && 
                (problem->bound_type[j] == BoundaryType::Upper || 
                problem->bound_type[j] == BoundaryType::Boxed)) ||
                problem->bound_type[j] == BoundaryType::Free)
                F.push_back(i);
        }

        int q_idx, q;
        double theta = INF;
        while (F.size() && delta >= 0)
        {
            int it = 0;
            q_idx = F[0];
            q = non_basis_indexes[q_idx];
            theta = (d[q] / tmp_alpha_p[q_idx]);
            for (auto i : F)
            {   
                int j = non_basis_indexes[i];
                double theta_tmp = d[j] / tmp_alpha_p[i];
                if (theta_tmp < theta || ((fabs(theta_tmp - theta) < EPS_Z && distrib(gen) == 1)))
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
            problem->solution.solved = false;
            problem->solution.message = "dual unbounded";
            break;
        }

        // (Step 6) FTran
        solveLinSys(problem->A(q), alpha_q, false);

        // (Step 7) Basis change and update
        // Update d accoprding to BRFT
        IndexVector infeas_idx;
        double delta_z = 0;

        for (int i = 0; i < non_basis_indexes.size(); i++)
        {
            int j = non_basis_indexes[i];
            d[j] = d[j] - theta * alpha_p[i];  
            if (problem->bound_type[j] == BoundaryType::Boxed)
            {
            
                if (fabs(x[j] - problem->lower_bound[j]) < EPS_Z && d[j] < -EPS_D)
                {
                    infeas_idx.push_back(j);
                    column_change += problem->A(j) * (problem->upper_bound[j] - problem->lower_bound[j]);
                    delta_z += problem->costs[j] * (problem->upper_bound[j] - problem->lower_bound[j]);
                }  
                else if (fabs(x[j] - problem->upper_bound[j]) < EPS_Z && d[j] > EPS_D)
                {
                    infeas_idx.push_back(j);
                    column_change -= problem->A(j) * (problem->upper_bound[j] - problem->lower_bound[j]);
                    delta_z -= problem->costs[j] * (problem->upper_bound[j] - problem->lower_bound[j]);
                }   
            }          
        }
        d[p] = -theta;
        d[q] = 0;

        if (!infeas_idx.size())
        {
            solveLinSys(column_change, delta_xB, false);
            x.setValues(x(basis_indexes) - delta_xB, basis_indexes);
            for (int i = 0; i < basis_size; i++)
            {
                int j = basis_indexes[i];
                delta_z -= problem->costs[j] * delta_xB[i];
            }    
        }
        obj_func_val += delta_z;

        // Update B and xB, DSE weights
        double theta_P = delta / alpha_q[p_idx];
        solveLinSys(rho, tau, false);
        for (int i = 0; i < basis_size; i++)
        {
            int j = basis_indexes[i];
            x[j] = x[j] - theta_P * alpha_q[i];  
            new_eta_matrix[i] = (i != p_idx) ? -alpha_q[i] / alpha_q[p_idx] : 1 / alpha_q[p_idx]; 
            beta[i] = (i != p_idx) ? beta[i] - 2 * alpha_q[i] / alpha_q[p_idx] * tau[i]  + pow(alpha_q[i] / alpha_q[p_idx], 2) * beta[p_idx] : beta[i]; 
        }  
        beta[p_idx] = beta[p_idx] / pow(alpha_q[p_idx], 2);
        x[q] = x[q] + theta_P;

        // Update basis
        basis_indexes[p_idx] =  q;
        non_basis_indexes[q_idx] = p;

        B_eta_repr.push_back(EtaMatrix(new_eta_matrix, p_idx));

        // Flip bounds
        for (auto j : infeas_idx)
        {
            if (fabs(x[j] - problem->lower_bound[j]) < EPS_Z)
                x[j] = problem->upper_bound[j];
            else 
                x[j] = problem->lower_bound[j]; 
        }
        obj_func_val += theta * delta;

        #ifdef DEBUG
            std::cout << iteration << " : Z = "<< obj_func_val  << " delta_z = " << delta << " p_info:" << p << " q_info:" << q  << std::endl;
        #endif

        if (fabs(theta) < EPS_A) cycle_num += 1;
    }
    std::cout << "iterations = " << iteration << std::endl;
    return problem->solution.solved;
}

