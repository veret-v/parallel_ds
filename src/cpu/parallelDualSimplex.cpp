#include "parallelDualSimplex.hpp"

//----------------------------------------------------------------------------------------
// Init params, setting data according to index arrays
//----------------------------------------------------------------------------------------
void ParallelDualSimplex::initDualSimplex()
{
    x = ValuesVector(problem->problem_size);
    d = ValuesVector(problem->problem_size);    
    B = problem->A(basis_indexes);
    B_eta_repr.clear();
    linalg::PFIdecompose(B, B_eta_repr);
}


//----------------------------------------------------------------------------------------
// Convert string to exiting methods for presolver
//----------------------------------------------------------------------------------------
PresolverMethods ParallelDualSimplex::stringToPreSolverMethod(
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
SolverMethods ParallelDualSimplex::stringToSolverMethod(
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
ParallelDualSimplex::Phase1OutStatus ParallelDualSimplex::callPresolver(const PresolverMethods method)
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
bool ParallelDualSimplex::callDualSolver(const SolverMethods method)
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
bool ParallelDualSimplex::callPrimalSolver()
{
   return true;
}


//----------------------------------------------------------------------------------------
// Initialize weights for DSE according to
// John J. Forrest and Donald Goldfarb. Steepest-edge simplex algorithms for
// linear programming. Math. Program., 57(3):341–374, 1992.
//----------------------------------------------------------------------------------------
ValuesVector ParallelDualSimplex::initBetaWeights()
{
    ValuesVector beta(problem->constraints_size);
    for (int i = 0; i < problem->constraints_size; i++)
        beta[i] = linalg::PFIsolve(B_eta_repr, linalg::unit(problem->constraints_size, i), true).norm();
    return beta;
}


//----------------------------------------------------------------------------------------
// Phase 1 method for finding dual fesaible basis, based on article:
// E. Kostina. The long step rule in the bounded-variable dual simplex method:
// Numerical experiments. Mathematical Methods of Operations Research, 55:413–
// 429, 2002.
//----------------------------------------------------------------------------------------
ParallelDualSimplex::Phase1OutStatus ParallelDualSimplex::minimizeDualInfeasibility()
{
    // (Step 1) Initialization
    Phase1OutStatus status;

    ValuesVector y(problem->constraints_size);
    ValuesVector rho(problem->constraints_size);
    ValuesVector f(problem->constraints_size);
    ValuesVector columns_change(problem->constraints_size);
    ValuesVector beta(problem->constraints_size);
    ValuesVector alpha(non_basis_size);
    ValuesVector alpha_q;
    ValuesVector alpha_tmp(non_basis_size);
    ValuesVector f_tmp(problem->constraints_size);
    ValuesVector new_eta_matrix(problem->constraints_size);
    ValuesVector tau(problem->constraints_size);

    IndexVector inf_u_indexes;
    IndexVector inf_l_indexes;
    IndexVector inf_f_indexes;
    IndexVector stub_index;

    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> distrib(0, 1);
    std::uniform_int_distribution<> non_basis_rand(0, non_basis_size);
    
    y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
    problem->A.dotUpdate(
        y, stub_index, 
        problem->costs, 
        d, -1, 1, 
        non_basis_indexes, 
        SpmvOptions::SET_UPDATE_T
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

        if (problem->bound_type[j] == BoundaryType::Free && d[j] < -EPS_D)
            inf_f_indexes.push_back(i);
    }

    obj_func_val = 0;
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

    f = linalg::PFIsolve(B_eta_repr, columns_change, false);
    
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
            y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
            problem->A.dotUpdate(
                y, stub_index, 
                problem->costs, 
                d, -1, 1, 
                non_basis_indexes, 
                SpmvOptions::SET_UPDATE_T
            );
        }

        // if (cycle_num > RESTART_CYCLE)
        // {
        //     status = Phase1OutStatus::NeedRestart;
        //     break;
        // }
        
        // (Step 2) Pricing
        int p, p_idx;
        double max_weight = 0;
        for (int i = 0; i < problem->constraints_size; i++)
        {
            int j = basis_indexes[i];
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
        rho = linalg::PFIsolve(B_eta_repr, linalg::unit(problem->constraints_size, p_idx), true);
        
        // (Step 4) Pivot row
        problem->A.dotUpdate(
            rho, stub_index, 
            problem->costs, 
            alpha, 1, 0, 
            non_basis_indexes, 
            SpmvOptions::FULL_UPDATE_T
        );

        // (Step 5) Ratio Test
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
        for (int i = 0; i < non_basis_indexes.size(); i++)
        {
            int j = non_basis_indexes[i];
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
        
        int q_idx = F[0];
        int q = non_basis_indexes[q_idx];;
        double theta = d[q] / alpha_tmp[q_idx];
        for (auto i : F)
        {   
            int j = non_basis_indexes[i];
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
        alpha_q = linalg::PFIsolve(B_eta_repr, problem->A(q), false);

        if (iteration % REFACT_FREQ == 0)
        {
            B_eta_repr.clear();
            linalg::PFIdecompose(B, B_eta_repr);
        }

        // (Step 7) Basis change and update
        obj_func_val = obj_func_val - theta * f[p_idx];
        double step = theta * f[p_idx];
        for (int i = 0; i < non_basis_indexes.size(); i++)
        {
            int j = non_basis_indexes[i];
            d[j] -= theta * alpha[i];    
        }
        d[p] = -theta;
        d[q] = 0;
        
        tau = linalg::PFIsolve(B_eta_repr, rho, false);
        for (int i = 0; i < problem->constraints_size; i++)
        {
            int j = basis_indexes[i];
            f[i] = (i != p_idx) ? f[i] - alpha_q[i] / alpha_q[p_idx] * f[p_idx] : f[p_idx];
            new_eta_matrix[i] = (i != p_idx) ? -alpha_q[i] / alpha_q[p_idx] : 1 / alpha_q[p_idx]; 
            beta[i] = (i != p_idx) ? beta[i] - 2 * alpha_q[i] / alpha_q[p_idx] * tau[i]  + pow(alpha_q[i] / alpha_q[p_idx], 2) * beta[p_idx] : beta[i]; 
        }  
        beta[p_idx] = beta[p_idx] / pow(alpha_q[p_idx], 2);
        f[p_idx] = f[p_idx] / alpha_q[p_idx];
        
        basis_indexes[p_idx] =  q;
        non_basis_indexes[q_idx] = p;
    
        B_eta_repr.push_back(EtaMatrix(new_eta_matrix, p_idx));

        inf_f_indexes.clear();
        for (int i = 0; i < non_basis_size; i++)
        {
            int j = non_basis_indexes[i];
            if (problem->bound_type[j] == BoundaryType::Free && d[j] < 0)
                inf_f_indexes.push_back(i);
        }

        calcDualInfeasible();
        if (fabs(theta) < EPS_A) 
            cycle_num += 1;
        else
            cycle_num = 0;

        #ifdef DEBUG
        std::cout << iteration << " : Z = "<< obj_func_val << " inf_num = " << counterDualInfeasible() << std::endl;  
        #endif
        
    }
    return status;
}


//----------------------------------------------------------------------------------------
// Phase 2 simple method for finding solution, based on article:
// Dipl. Inform. Achim Koberstein. The Dual Simplex Method, Techniques for a fast and 
// stable implementation: for a fast and stable implementation, November 2005
//----------------------------------------------------------------------------------------
bool ParallelDualSimplex::elaboratedMethod()
{
    // (Step 1) Initialization
    int max_candidates_num = omp_get_num_threads();

    ValuesVector y(problem->constraints_size);
    ValuesVector beta(problem->constraints_size);
    ValuesVector candidate_weights(max_candidates_num);
    ValuesVector rho(problem->constraints_size);
    ValuesVector alpha_p(non_basis_size);
    ValuesVector tmp_alpha_p(non_basis_size);
    ValuesVector alpha_q(basis_size);
    ValuesVector column_change(problem->constraints_size);
    ValuesVector delta_xB(basis_size);
    ValuesVector tau(problem->constraints_size);
    ValuesVector new_eta_matrix(problem->constraints_size);

    IndexVector pivot_candidats(max_candidates_num);
    IndexVector stub_index;
 
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> distrib(0, 1);
    std::uniform_int_distribution<> non_basis_rand(0, non_basis_size);
     
    problem->A.dotUpdate(
        x, non_basis_indexes, 
        problem->RHS, 
        problem->RHS, -1, 1, 
        non_basis_indexes, 
        SpmvOptions::FULL_UPDATE
    );
    x.setValues(linalg::PFIsolve(B_eta_repr, problem->RHS, false), basis_indexes);
    y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
    problem->A.dotUpdate(
        y, non_basis_indexes, 
        problem->costs, 
        d, -1, 1, 
        non_basis_indexes, 
        SpmvOptions::SET_UPDATE_T
    );
    obj_func_val = problem->costs.dot(x);

    int iteration = 0;
    int cycle_num = 0;


    while (true)
    {
        if (!perturbed && cycle_num > MAX_CYCLE) 
        {
            perturbCosts();
            y = linalg::PFIsolve(B_eta_repr, problem->costs(basis_indexes), true);
            problem->A.dotUpdate(
                y, non_basis_indexes, 
                problem->costs, 
                d, -1, 1, 
                non_basis_indexes, 
                SpmvOptions::SET_UPDATE_T
            );
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
        
        for (int i = 0; i < problem->constraints_size; i++)
        {
            int j = basis_indexes[i];
            double delta_tmp = 0;
            bool is_lower_tmp; 
           
            bool _ = setDelta(j, delta_tmp, is_lower_tmp);

            double weight_tmp = pow(delta_tmp, 2) / beta[i];
            for (int k = 0; k < max_candidates_num; k++)
            {
                if (weight_tmp > candidate_weights[k] && delta_tmp != 0)
                {
                    pivot_candidats[k] = i;
                    candidate_weights[k] = weight_tmp;
                    break;
                }
                
            }
        }

        int candidates_num = candidate_weights.countNonZero();
        int dual_unbound_counter = 0;
        IndexVector entering_candidates(candidates_num);
        std::vector<double> Z_deltas(candidates_num);
        std::vector<ValuesVector> candidate_betas(candidates_num);
        std::vector<ValuesVector> candidate_x(candidates_num);
        std::vector<ValuesVector> candidate_d(candidates_num);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int cand_id = 0; cand_id < candidates_num; cand_id++)
        {
            int p_idx = pivot_candidats[cand_id];
            int p = basis_indexes[p_idx];
            bool is_lower;
            double delta;
            bool _ = setDelta(p, delta, is_lower);
            
            // (Step 3) BTran
            rho = linalg::PFIsolve(B_eta_repr, linalg::unit(problem->constraints_size, p_idx), true);

            // (Step 4) Pivot row
            problem->A.dotUpdate(
                rho, stub_index, 
                problem->costs, 
                alpha_p, 1, 0, 
                non_basis_indexes, 
                SpmvOptions::FULL_UPDATE_T
            );

            // (Step 5) Ratio Test
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

            int q_idx, q;
            double theta;
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
                dual_unbound_counter += 1;
                continue;
            }

            // (Step 6) FTran
            alpha_q = linalg::PFIsolve(B_eta_repr, problem->A(q), false);

            // (Step 7) Basis change and update

            // Update d accoprding to BRFT
            IndexVector infeas_idx;
            double delta_z = 0;
            ValuesVector updated_d = d;
            for (int i = 0; i < non_basis_indexes.size(); i++)
            {
                int j = non_basis_indexes[i];
                updated_d[j] = d[j] - theta * alpha_p[i];  
                if (problem->bound_type[j] == BoundaryType::Boxed)
                {
                
                    if (x[j] == problem->lower_bound[j] && updated_d[j] < 0)
                    {
                        infeas_idx.push_back(j);
                        column_change += problem->A(j) * (problem->upper_bound[j] - problem->lower_bound[j]);
                        delta_z += problem->costs[j] * (problem->upper_bound[j] - problem->lower_bound[j]);
                    }  
                    else if (x[j] == problem->upper_bound[j] && updated_d[j] > 0)
                    {
                        infeas_idx.push_back(j);
                        column_change -= problem->A(j) * (problem->upper_bound[j] - problem->lower_bound[j]);
                        delta_z -= problem->costs[j] * (problem->upper_bound[j] - problem->lower_bound[j]);
                    }   
                }          
            }
            updated_d[p] = -theta;
            updated_d[q] = 0;

            ValuesVector updated_x = x;
            if (!infeas_idx.size())
            {
                delta_xB = linalg::PFIsolve(B_eta_repr, column_change, false);
                updated_x.setValues(updated_x(basis_indexes) - delta_xB, basis_indexes);
                for (int i = 0; i < problem->constraints_size; i++)
                {
                    int j = basis_indexes[i];
                    delta_z -= problem->costs[j] * delta_xB[i];
                }    
            }

            // Update xB, DSE weights
            double theta_P = delta / alpha_q[p_idx];
            ValuesVector updated_beta = beta;
            tau = linalg::PFIsolve(B_eta_repr, rho, false);
            for (int i = 0; i < problem->constraints_size; i++)
            {
                int j = basis_indexes[i];
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
        

        int opt_cand_id = std::distance(Z_deltas.begin(), std::max_element(Z_deltas.begin(), Z_deltas.end()));
        int opt_p_idx   = pivot_candidats[opt_cand_id];
        int opt_p       = basis_indexes[opt_p_idx];
        int opt_q_idx   = entering_candidates[opt_cand_id];
        int opt_q       = non_basis_indexes[opt_q_idx];
        x =    candidate_x[opt_cand_id];
        beta = candidate_betas[opt_cand_id];
        d =    candidate_d[opt_cand_id];

        // Update basis
        ValuesVector opt_alpha_q;
        opt_alpha_q = linalg::PFIsolve(B_eta_repr, problem->A(opt_q), false);

        for (int i = 0; i < problem->constraints_size; i++)
            new_eta_matrix[i] = (i != opt_p_idx) ? -opt_alpha_q[i] / opt_alpha_q[opt_p_idx] : 1 / opt_alpha_q[opt_p_idx]; 
        
        basis_indexes[opt_p_idx] =  opt_q;
        non_basis_indexes[opt_q_idx] = opt_p;
        B_eta_repr.push_back(EtaMatrix(new_eta_matrix, opt_p_idx));

        obj_func_val += Z_deltas[opt_cand_id];
        iteration += 1;
        if (fabs(Z_deltas[opt_cand_id]) < EPS_A) cycle_num += 1;
    }
    std::cout << "iterations = " << iteration << std::endl;
    return solution.solved;
}


