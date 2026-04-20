#include "cudaDualSimplex.hpp"


CudaDualSimplex::~CudaDualSimplex()
{
    cusparseDestroy(sp_handle);
    cublasDestroy(cu_handle);

    cudssDataDestroy(cudss_handle, cudss_data);
    cudssConfigDestroy(cudss_config);
    cudssDestroy(cudss_handle);

    cudssDataDestroy(cudss_handle_T, cudss_data_T);
    cudssConfigDestroy(cudss_config_T);
    cudssDestroy(cudss_handle_T);

    cudaStreamSynchronize(stream);
}

//----------------------------------------------------------------------------------------
// Init params, setting data according to index arrays
//----------------------------------------------------------------------------------------
void CudaDualSimplex::initDualSimplex()
{
    std::cout << "Solver initialization : basis columns selected" << std::endl;

    x = ValuesVector(problem->problem_size);
    d = ValuesVector(problem->problem_size);   

    // начальная базисная матрица является единичной, так как в 
    // качсетве базиса берутся basis_size последних колонок матрицы A, 
    // а они в свою очередь созданы добавление slack-variables
    B.initI(basis_size); 
    cudaStreamCreate(&stream);

    cusparseCreate(&sp_handle);
    cublasCreate(&cu_handle);
    cusparseSetStream(sp_handle, stream);
    cublasSetStream(cu_handle, stream);

    cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;    
    cudssAlgType_t matching_alg = CUDSS_ALG_DEFAULT; // matching with scaling, same as CUDSS_ALG_5
    int ione = 1;

    /*initalization backend for A LU factorization*/
    cudssCreate(&cudss_handle);
    cudssSetStream(cudss_handle, stream);
    cudssDataCreate(cudss_handle, &cudss_data);
    cudssConfigCreate(&cudss_config);

    cudssConfigSet(cudss_config, CUDSS_CONFIG_REORDERING_ALG,
                         &reorder_alg, sizeof(cudssAlgType_t));
    cudssConfigSet(cudss_config, CUDSS_CONFIG_USE_MATCHING,
                         &ione, sizeof(int));
    cudssConfigSet(cudss_config, CUDSS_CONFIG_MATCHING_ALG,
                         &matching_alg, sizeof(cudssAlgType_t));

    /*initalization backend for AT LU factorization*/
    cudssCreate(&cudss_handle_T);
    cudssSetStream(cudss_handle_T, stream);
    cudssDataCreate(cudss_handle_T, &cudss_data_T);
    cudssConfigCreate(&cudss_config_T);

    cudssConfigSet(cudss_config_T, CUDSS_CONFIG_REORDERING_ALG,
                         &reorder_alg, sizeof(cudssAlgType_t));
    cudssConfigSet(cudss_config_T, CUDSS_CONFIG_USE_MATCHING,
                         &ione, sizeof(int));
    cudssConfigSet(cudss_config_T, CUDSS_CONFIG_MATCHING_ALG,
                         &matching_alg, sizeof(cudssAlgType_t));

           
    problem->A.genCsc(sp_handle);

    B.createDescr();
    B.LUdecompose(
        cudss_handle, cudss_config, cudss_data,
        cudss_handle_T, cudss_config_T, cudss_data_T
    );

    std::cout << "Solver initialization : attributes setted" << std::endl; 
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
Phase1OutStatus CudaDualSimplex::callPresolver(const PresolverMethods method)
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
    beta.resize(basis_size);
    for (int i = 0; i < basis_size; i++)
        beta[i] = 1;
}

//----------------------------------------------------------------------------------------
// Solve system of linear equations by LU factorization and PFI updates.
// PBQ = LU -> B = P^-1LUQ^-1, 
// after column swap B0 -> B1 = E1 * B0 = E1 * P^-1LUQ^-1 - this is LU + PFI
//----------------------------------------------------------------------------------------
void CudaDualSimplex::solveLinSys(
    const bool transpose, 
    const CudaDataDenseVector& rhs, 
    CudaDataDenseVector& sol
)
{
    if (transpose)
    {
        pfi_factor.applyPFI(
            cu_handle, rhs, 
            sol, transpose
        );
        B.solve(
            cudss_handle_T, cudss_config_T, 
            cudss_data_T, rhs, sol, transpose
        );
    }
    else
    {
        B.solve(
            cudss_handle, cudss_config, 
            cudss_data, rhs, sol, transpose
        );
        pfi_factor.applyPFI(
            cu_handle, rhs, 
            sol, transpose
        );
    }
}

//----------------------------------------------------------------------------------------
// Phase 1 method for finding dual fesaible basis, based on article:
// E. Kostina. The long step rule in the bounded-variable dual simplex method:
// Numerical experiments. Mathematical Methods of Operations Research, 55:413–
// 429, 2002.
//----------------------------------------------------------------------------------------
Phase1OutStatus CudaDualSimplex::minimizeDualInfeasibility()
{
    // (Step 1) Initialization
    Phase1OutStatus     status;

    CudaIndexVector     stub_index;
    CudaIndexVector     stub_val;

    CudaDataDenseVector columns_change(basis_size);
    CudaDataDenseVector rho(basis_size);
    CudaDataDenseVector y(basis_size);
    CudaDataDenseVector rhs(basis_size);
    CudaDataDenseVector alpha(non_basis_size);
    CudaDataDenseVector alpha_tmp(non_basis_size);
    CudaDataDenseVector f(basis_size);
    CudaDataDenseVector alpha_q(basis_size);
    CudaDataDenseVector tau(basis_size);
    CudaDataDenseVector new_eta_matrix(basis_size);

    IndexVector         inf_u_indexes;
    IndexVector         inf_l_indexes;
    IndexVector         inf_f_indexes;

    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> distrib(0, 1);
    std::uniform_int_distribution<> non_basis_rand(0, non_basis_size);


    rhs.updateByPartialVec(problem->costs, basis_indexes);
    rhs.updateDeviceMem();
    solveLinSys(true, rhs, y);

    problem->costs.updateDeviceMem();
    non_basis_indexes.updateDeviceMem();
    problem->A.dotUpdate(
        sp_handle,
        y, problem->costs, 
        d, -1, 1, 
        non_basis_indexes, 
        SpmvOptions::UPDATE_T,
        true
    );
    d.updateHostMem();

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

    for (auto i : inf_l_indexes) obj_func_val += d[i];
    problem->A.addSparseCol(sp_handle, columns_change, inf_l_indexes, 1);

    for (auto i : inf_u_indexes) obj_func_val += d[i];
    problem->A.addSparseCol(sp_handle, columns_change, inf_u_indexes, -1);

    solveLinSys(false, columns_change, f);
    f.updateHostMem();

    initBetaWeights();

    int iteration = 0;
    int cycle_num = 0;

    std::unordered_set<int> blocked_p;
    while (true)
    {
        iteration += 1;
       
        if (!perturbed && cycle_num > MAX_CYCLE) 
        {
            perturbCosts();
            problem->costs.updateDeviceMem();

            rhs.updateByPartialVec(problem->costs, basis_indexes);
            rhs.updateDeviceMem();
            solveLinSys(true, rhs, y);

             problem->A.dotUpdate(
                sp_handle,
                y, problem->costs, 
                d, -1, 1, 
                non_basis_indexes, 
                SpmvOptions::UPDATE_T,
                true
            );
            d.updateHostMem();
        }

        if (cycle_num > NEED_RESTART || 
            obj_func_val > INF || std::isnan(obj_func_val))
        {
            status = Phase1OutStatus::NeedRestart;
            break;
        }

        if (iteration % REFACT_FREQ == 0)
        {
            pfi_factor.resetPFI();
            B.resetData(sp_handle, problem->A, basis_indexes);
            B.LUdecompose(
                cudss_handle, cudss_config, cudss_data,
                cudss_handle, cudss_config, cudss_data
            );
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
        rhs.initUnitVec(p_idx);
        rhs.updateDeviceMem();
        solveLinSys(true, rhs, rho);
        
        // (Step 4) Pivot row
        problem->A.dotUpdate(
            sp_handle,
            rho, problem->costs, 
            alpha, 1, 0, 
            non_basis_indexes, 
            SpmvOptions::UPDATE_T,
            false
        );
        alpha.updateHostMem();
       
        // (Step 5) Ratio Test
        if (f[p_idx] > 0) alpha.multiplyHostData(-1);
        IndexVector F;
        for (int i = 0; i < non_basis_indexes.getSize(); i++)
        {
            int j = non_basis_indexes[i];
            if (((d[j] >= 0 && alpha[i] > EPS_A) ||
                (d[j] <= 0 && alpha[i] < -EPS_A)))
                F.push_back(i);                
        }

        if (!F.size()) 
        {
            blocked_p.insert(p);
            if (blocked_p.size() > RESTART_SIZE)
            {
                status = Phase1OutStatus::NeedRestart;
                break; 
            }
            continue;
        }

        if (blocked_p.size())
            blocked_p.clear();
        
        int q_idx, q;
        double theta = INF;
        for (auto i : F)
        {   
            int j = non_basis_indexes[i];
            double theta_tmp = d[j] / alpha[i];
            if (theta_tmp < theta || (fabs(theta_tmp - theta) < EPS_Z && distrib(gen) == 1)) 
            {
                theta = theta_tmp;
                q = j;
                q_idx = i;
            } 
            else if (theta_tmp < theta + EPS_Z)
            {
                if (fabs(alpha[i]) > fabs(alpha[q_idx]))
                {
                    theta = theta_tmp;
                    q = j;
                    q_idx = i;
                }
            }
        }

        if (f[p_idx] > 0) alpha.multiplyHostData(-1);
        theta = d[q] / alpha[q_idx]; 
        
        // (Step 6) FTran
        rhs.updateHostMem();
        rhs.show();
        problem->A.show();
        exit();
        problem->A.getColumn(sp_handle, q, rhs);
        rhs.updateHostMem();
        rhs.show();
        
        solveLinSys(false, rhs, alpha_q);
        alpha_q.updateHostMem();
        alpha_q.show();

        // (Step 7) Basis change and update
        double infisib_corr = 0;
        if ((problem->bound_type[q] == BoundaryType::Upper || 
            problem->bound_type[q] == BoundaryType::Free) && d[q] > EPS_D)  
            infisib_corr = 1;
        else if ((problem->bound_type[q] == BoundaryType::Lower || 
            problem->bound_type[q] == BoundaryType::Free) && d[q] < -EPS_D) 
            infisib_corr = -1;

        obj_func_val = obj_func_val - theta * f[p_idx];
        for (int i = 0; i < non_basis_indexes.getSize(); i++)
        {
            int j = non_basis_indexes[i];
            d[j] -= theta * alpha[i];    
        }
        d[p] = -theta;
        d[q] = 0;
        
        solveLinSys(false, rho, tau);
        tau.updateHostMem();
        // todo : оформить цикл в ядро
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
        
        basis_indexes.update(p_idx, q);
        non_basis_indexes.update(q_idx, p);

        pfi_factor.addEtaMatrix(p_idx, new_eta_matrix);

        cycle_num = (fabs(theta) < EPS_A) ? cycle_num + 1 : 0;

        #ifdef DEBUG
            std::cout << iteration << " : Z = "<< obj_func_val << " inf_num = " << counterDualInfeasible() << " p_info:" << p << " q_info:" << q << " f:" << f[p_idx] << " beta:" << beta[p_idx] << " alpha_q:" << alpha_q[p_idx]<< std::endl;  
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
    CudaIndexVector     stub_index;
    CudaIndexVector     stub_val;

    CudaDataDenseVector y(basis_size);
    CudaDataDenseVector rhs(basis_size);
    CudaDataDenseVector buff_sol(basis_size);
    CudaDataDenseVector rho(basis_size);
    CudaDataDenseVector alpha_p(non_basis_size);
    CudaDataDenseVector tmp_alpha_p(non_basis_size);
    CudaDataDenseVector alpha_q(basis_size);
    CudaDataDenseVector tau(basis_size);
    CudaDataDenseVector column_change(basis_size);
    CudaDataDenseVector delta_xB(basis_size);
    CudaDataDenseVector new_eta_matrix(basis_size);

    std::vector<double> difference_bounds = problem->upper_bound - problem->lower_bound;
        
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> distrib(0, 1);
    std::uniform_int_distribution<> non_basis_rand(0, non_basis_size);

     problem->A.dotUpdate(
        sp_handle,
        x, problem->RHS, 
        problem->RHS, -1, 1, 
        non_basis_indexes, 
        SpmvOptions::UPDATE,
        true
    );
    problem->RHS.updateHostMem();

    solveLinSys(true, problem->RHS, buff_sol);
    buff_sol.updateHostMem();
    x.updateByPartialVec(buff_sol, basis_indexes);

    rhs.updateByPartialVec(problem->costs, basis_indexes);
    rhs.updateDeviceMem();
    solveLinSys(true, rhs, y);

     problem->A.dotUpdate(
        sp_handle,
        y, problem->costs, 
        d, -1, 1, 
        non_basis_indexes, 
        SpmvOptions::UPDATE_T,
        true
    );
    d.updateHostMem();

    x.updateDeviceMem();
    obj_func_val = problem->costs.dot(cu_handle, x);

    int iteration = 0;
    int cycle_num = 0;

    while (true)
    {
        if (!perturbed && cycle_num > MAX_CYCLE) 
        {
            perturbCosts();
            
            rhs.updateByPartialVec(problem->costs, basis_indexes);
            rhs.updateDeviceMem();
            solveLinSys(true, rhs, y);

            problem->A.dotUpdate(
                sp_handle,
                y, problem->costs, 
                d, -1, 1, 
                non_basis_indexes, 
                SpmvOptions::UPDATE_T,
                true
            );
            d.updateHostMem();

            x.updateDeviceMem();
            obj_func_val = problem->costs.dot(cu_handle, x);
        }
        #ifdef DEBUG
        std::cout << iteration << " : Z = "<< obj_func_val << std::endl;
        #endif
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
        rhs.initUnitVec(p_idx);
        rhs.updateDeviceMem();
        solveLinSys(true, rhs, rho);

        // (Step 4) Pivot row
        problem->A.dotUpdate(
            sp_handle,
            rho, problem->costs, 
            alpha_p, 1, 0, 
            non_basis_indexes, 
            SpmvOptions::UPDATE_T,
            false
        );
        alpha_p.updateHostMem();

        // (Step 5) Ratio Test
        delta = fabs(delta);
        int sgn = (is_lower) ? -1 : 1;
        tmp_alpha_p = (is_lower) ? -alpha_p : alpha_p;

        IndexVector F;
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
                if (theta_tmp < theta || (fabs(theta_tmp - theta) < EPS_Z && distrib(gen) == 1))
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
        problem->A.getColumn(sp_handle, q_idx, rhs);
        rhs.updateDeviceMem();
        solveLinSys(false, rhs, alpha_q);
        alpha_q.updateHostMem();

        // (Step 7) Basis change and update

        // Update d accoprding to BRFT
        double delta_z = 0;
        IndexVector infeas_idx;

        for (int i = 0; i < non_basis_indexes.getSize(); i++)
        {
            int j = non_basis_indexes[i];
            d[j] = d[j] - theta * alpha_p[i];  
            if (problem->bound_type[j] == BoundaryType::Boxed)
            {
            
                if (x[j] == problem->lower_bound[j] && d[j] < 0)
                {
                    infeas_idx.push_back(j);
                    delta_z += problem->costs[j] * (problem->upper_bound[j] - problem->lower_bound[j]);
                }  
                else if (x[j] == problem->upper_bound[j] && d[j] > 0)
                {
                    infeas_idx.push_back(j);
                    delta_z -= problem->costs[j] * (problem->upper_bound[j] - problem->lower_bound[j]);
                }   
            }          
        }

        problem->A.addSparseCol(sp_handle, column_change, infeas_idx, difference_bounds); 
        d[p] = -theta;
        d[q] = 0;

        if (!infeas_idx.size())
        {
            column_change.updateDeviceMem();
            solveLinSys(false, column_change, delta_xB);

            rhs.updateByPartialVec(x, basis_indexes);
            rhs.axpyUpdate(cu_handle, delta_xB, -1);
            x.updateByPartialVec(rhs, basis_indexes);

            for (int i = 0; i < basis_size; i++)
            {
                int j = basis_indexes[i];
                delta_z -= problem->costs[j] * delta_xB[i];
            }    
        }

        obj_func_val += delta_z;

        // Update B and xB, DSE weights
        double theta_P = delta / alpha_q[p_idx];
        
        solveLinSys(false, rho, tau);
        tau.updateHostMem();

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
        basis_indexes.update(p_idx, q);
        non_basis_indexes.update(q_idx, p);
    
        pfi_factor.addEtaMatrix(p_idx, new_eta_matrix);

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
    return problem->solution.solved;
}

