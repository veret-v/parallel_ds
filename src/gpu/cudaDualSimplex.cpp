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
    //  problem->A.createDescr();

    B.createDescr();
    B.LUdecompose(
        cudss_handle, cudss_config, cudss_data,
        cudss_handle_T, cudss_config_T, cudss_data_T
    );
    pfi_factor.resetPFI();

    _timer = std::make_unique<TimeProfiler>(TimeProfiler());

    std::cout << "Solver initialization : attributes setted" << std::endl; 
}

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


bool CudaDualSimplex::callPrimalSolver()
{
   return true;
}


void CudaDualSimplex::solveLinSys(
    CudaDataDenseVector& rhs, 
    CudaDataDenseVector& sol,
    bool transpose
)
{
    if (transpose)
    {
        auto _prev_time = std::chrono::high_resolution_clock::now();
        pfi_factor.applyPFI(
            cu_handle, rhs, 
            sol, transpose
        );
        CudaDataDenseVector new_rhs = sol;
        auto _curr_time = std::chrono::high_resolution_clock::now();
        pfi += std::chrono::duration_cast<std::chrono::milliseconds>(_curr_time - _prev_time).count();

        _prev_time = std::chrono::high_resolution_clock::now();
        B.solve(
            cudss_handle_T, cudss_config_T, 
            cudss_data_T, new_rhs, sol, transpose
        );
        _curr_time = std::chrono::high_resolution_clock::now();
        ker += std::chrono::duration_cast<std::chrono::milliseconds>(_curr_time - _prev_time).count();

    }
    else
    {
        auto _prev_time = std::chrono::high_resolution_clock::now();
        B.solve(
            cudss_handle, cudss_config, 
            cudss_data, rhs, sol, transpose
        );
        auto _curr_time = std::chrono::high_resolution_clock::now();
        ker += std::chrono::duration_cast<std::chrono::milliseconds>(_curr_time - _prev_time).count();

        _prev_time = std::chrono::high_resolution_clock::now();
        CudaDataDenseVector new_rhs = sol;
        pfi_factor.applyPFI(
            cu_handle, new_rhs, 
            sol, transpose
        );
        _curr_time = std::chrono::high_resolution_clock::now();
        pfi += std::chrono::duration_cast<std::chrono::milliseconds>(_curr_time - _prev_time).count();

    }
}


void CudaDualSimplex::BTran(int p_idx, CudaDataDenseVector& rho)
{
    CudaDataDenseVector rhs(basis_size);
    rhs.initUnitVec(p_idx);
    rhs.updateDeviceMem();
    solveLinSys(rhs, rho, true);
}


void CudaDualSimplex::pivotRow(CudaDataDenseVector& rho, CudaDataDenseVector& alpha)
{
    problem->A.dotUpdate(
            sp_handle,
            rho, problem->costs, 
            alpha, 1, 0, 
            non_basis_indexes, 
            SpmvOptions::UPDATE_T,
            false
        );
    alpha.updateHostMem();
}


void CudaDualSimplex::FTran(int q, CudaDataDenseVector& alpha_q)
{
    CudaDataDenseVector rhs(basis_size);
    problem->A.getColumn(sp_handle, q, rhs);
    solveLinSys(rhs, alpha_q, false);
    alpha_q.updateHostMem();
}


void CudaDualSimplex::initReducedCosts()
{
    CudaDataDenseVector rhs(basis_size);
    CudaDataDenseVector y(basis_size);
    
    basis_indexes.updateDeviceMem();
    non_basis_indexes.updateDeviceMem();
    problem->costs.updateDeviceMem();

    rhs.updateByPartialVec(problem->costs, basis_indexes);
    rhs.updateDeviceMem();

    solveLinSys(rhs, y, true);

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


void CudaDualSimplex::initPhase1PricingVector(CudaDataDenseVector& f, IndexVector& inf_u_indexes, IndexVector& inf_l_indexes, double& Z)
{
    CudaDataDenseVector columns_change(basis_size);

    Z = 0;

    for (auto i : inf_l_indexes) Z += d[i];
    problem->A.addSparseCol(sp_handle, columns_change, inf_l_indexes, 1);

    for (auto i : inf_u_indexes) Z -= d[i];
    problem->A.addSparseCol(sp_handle, columns_change, inf_u_indexes, -1);

    solveLinSys(columns_change, f, false);
    f.updateHostMem();
}


void CudaDualSimplex::reFactorize()
{
    pfi_factor.resetPFI();
    B.resetData(sp_handle, problem->A, basis_indexes);
    B.createDescr();
    B.LUdecompose(
        cudss_handle, cudss_config, cudss_data,
        cudss_handle_T, cudss_config_T, cudss_data_T
    );
}


void CudaDualSimplex::simpleReducedCostsUpate(const CudaDataDenseVector& alpha, int p, int q, double theta)
{
    #pragma omp parallel for
    for (int i = 0; i < non_basis_indexes.getSize(); i++)
    {
        int j = non_basis_indexes[i];
        d[j] -= theta * alpha[i];    
    }
    d[p] = -theta;
    d[q] = 0;
}


void CudaDualSimplex::updateAndChangeBasis(
    CudaDataDenseVector& f, CudaDataDenseVector& rho, const CudaDataDenseVector& alpha_q, 
    int p_idx, int p, int q_idx, int q, double theta_P
) 
{
    CudaDataDenseVector tau(basis_size);
    CudaDataDenseVector new_eta_matrix(basis_size);

    _timer->startTimer();
    solveLinSys(rho, tau, false);
    tau.updateHostMem();
    _timer->stopTimer(ALgorithmPart::FtranBeta);

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
}


void CudaDualSimplex::phase1UpdateAndChangeBasis(
    CudaDataDenseVector& f, CudaDataDenseVector& rho, const CudaDataDenseVector& alpha_q, 
    int p_idx, int p, int q_idx, int q, double theta_P
) 
{
    CudaDataDenseVector tau(basis_size);
    CudaDataDenseVector new_eta_matrix(basis_size);

    solveLinSys(rho, tau, false);
    tau.updateHostMem();
    // todo : оформить цикл в ядро
    for (int i = 0; i < basis_size; i++)
    {
        int j = basis_indexes[i];
        f[i] = (i != p_idx) ? f[i] - alpha_q[i] * theta_P : f[p_idx];
        new_eta_matrix[i] = (i != p_idx) ? -alpha_q[i] / alpha_q[p_idx] : 1 / alpha_q[p_idx]; 
        beta[i] = (i != p_idx) ? beta[i] - 2 * alpha_q[i] / alpha_q[p_idx] * tau[i]  + pow(alpha_q[i] / alpha_q[p_idx], 2) * beta[p_idx] : beta[i]; 
    }  
    beta[p_idx] = beta[p_idx] / pow(alpha_q[p_idx], 2);

    basis_indexes.update(p_idx, q);
    non_basis_indexes.update(q_idx, p);
    
    pfi_factor.addEtaMatrix(p_idx, new_eta_matrix);
}


void CudaDualSimplex::dualSimplexInit()
{
    CudaDataDenseVector buff_sol(basis_size);
    CudaDataDenseVector rhs(basis_size);

    setPrimalVars();

    x.updateDeviceMem();
    problem->A.dotUpdate(
        sp_handle,
        x, problem->RHS, 
        rhs, -1, 1, 
        non_basis_indexes, 
        SpmvOptions::UPDATE,
        true
    );
    
    solveLinSys(rhs, buff_sol, false);
    buff_sol.updateHostMem();
    x.setValues(buff_sol, basis_indexes);
   
    initReducedCosts();
    x.updateDeviceMem();
    obj_func_val = problem->costs.dot(cu_handle, x);
}


bool CudaDualSimplex::callDualSolver()
{
    
    // (Step 1) Initialization
    CudaDataDenseVector rho(basis_size);
    CudaDataDenseVector alpha_p(non_basis_size);
    CudaDataDenseVector alpha_q(basis_size);

    std::vector<double> difference_bounds = std::move(problem->upper_bound - problem->lower_bound);
    
    _timer->startTimer();
    dualSimplexInit();
    _timer->stopTimer(ALgorithmPart::Init);

    int cycle_num = 0;

    // check that basis truly undounded and not because numeric error
    bool unbound_checked = false; 
    std::unordered_set<int> blocked_p;

    while (true)
    {
        iteration += 1;
        
        if (iteration % REFACT_FREQ == 0)
        {
            _timer->startTimer();
            reFactorize();
            _timer->stopTimer(ALgorithmPart::Factor);
        }

        if (iteration % NEED_RESTART == 0 && !checkDualFeasible())
        {
            std::cout << "-- Soft restart. Dual infeasiblity find." << std::endl;  
            _timer->startTimer();
            reFactorize();
            minimizeDualInfeasibility();
            dualSimplexInit();
            _timer->stopTimer(ALgorithmPart::RestoreProc);
        }

        if (!perturbed && cycle_num > MAX_CYCLE) 
        {
            std::cout << "-- Soft restart. Perturb costs." << std::endl; 
            _timer->startTimer();
            perturbCosts();
            reFactorize();
            dualSimplexInit();
            _timer->stopTimer(ALgorithmPart::RestoreProc);
        }
        
        if (checkPrimalFeasible())
        {
            problem->solution.solved = true;
            problem->solution.message = "optimal solution";
            break;
        }
        
        // (Step 2) Pricing
        auto [p, p_idx, delta, is_lower] = DSEPricing(blocked_p);
        if (p < 0 || p_idx < 0)
        {
            std::cout << "-- Soft restart. No p find." << std::endl;  
            reFactorize();
            initBetaWeights(false);
            dualSimplexInit();
            blocked_p.clear();
            continue;
        }
        
        // (Step 3) BTran
        _timer->startTimer();
        BTran(p_idx, rho);
        _timer->stopTimer(ALgorithmPart::Btran);

        // (Step 4) Pivot row
        _timer->startTimer();
        pivotRow(rho, alpha_p);
        _timer->stopTimer(ALgorithmPart::PivotRow);

        // (Step 5) Ratio Test
        _timer->startTimer();
        delta = fabs(delta);
        int sgn = (is_lower) ? -1 : 1;
        if (is_lower) alpha_p.multiplyData(-1);

        IndexVector F; F.reserve(non_basis_size);  
        int cnt = 0;
        for (int i = 0; i < non_basis_size; ++i) {
            bool cond = false;
            double a = alpha_p[i];
            if (a > EPS_A) {
                int j = non_basis_indexes[i];
                double xj = x[j];
                if (fabs(xj - problem->lower_bound[j]) < EPS_Z &&
                    (problem->bound_type[j] == BoundaryType::Lower || problem->bound_type[j] == BoundaryType::Boxed))
                    cond = true;
            } else if (a < -EPS_A) {
                int j = non_basis_indexes[i];
                double xj = x[j];
                if (fabs(xj - problem->upper_bound[j]) < EPS_Z &&
                    (problem->bound_type[j] == BoundaryType::Upper || problem->bound_type[j] == BoundaryType::Boxed))
                    cond = true;
            } else if (a == 0) { // учтём Free
                int j = non_basis_indexes[i];
                if (problem->bound_type[j] == BoundaryType::Free && fabs(x[j]) < EPS_Z)
                    cond = true;
            }
            if (cond) F.push_back(i);
        }
        _timer->stopTimer(ALgorithmPart::RatioTestPart1);

        // check that problem is really uboundedd, not because of numeric error
        if (!F.size())
        {
            if (unbound_checked)
            {
                problem->solution.solved = false;
                problem->solution.message = "dual unbounded";
                break;
            }
            else
            {
                _timer->startTimer();
                std::cout << "-- Soft restart. Check unboundness." << std::endl;  
                unbound_checked = true;

                reFactorize();
                initBetaWeights(false);
                minimizeDualInfeasibility();
                dualSimplexInit();
                _timer->stopTimer(ALgorithmPart::RestoreProc);

                continue;
            }
        }
        else
        {
            if (unbound_checked)
            {
                unbound_checked = false;
            }
        }

        _timer->startTimer();
        int q_idx, q;
        double theta = INF;
        while (F.size())
        {
            std::tie(q, q_idx) = simpleRatioTest(F, alpha_p);
            if (problem->bound_type[q] == BoundaryType::Boxed)
            {
                if (delta - (problem->upper_bound[q] - problem->lower_bound[q]) * fabs(alpha_p[q_idx]) <= 0)
                {
                    break;
                }
                delta -= (problem->upper_bound[q] - problem->lower_bound[q]) * fabs(alpha_p[q_idx]);
                F.erase(std::remove(F.begin(), F.end(), q_idx), F.end());
            }
            else
            {
                break;
            }
        }
        if (is_lower) alpha_p.multiplyData(-1);
        theta = d[q] / alpha_p[q_idx];
        delta = sgn * delta;
        _timer->stopTimer(ALgorithmPart::RatioTestPart2);

        // (Step 6) FTran
        _timer->startTimer();
        FTran(q, alpha_q);
        _timer->stopTimer(ALgorithmPart::Ftran);

        // (Step 7) Basis change and update
        // Update d accoprding to BRFT
        _timer->startTimer();
        IndexVector low_infeas_idx, up_infeas_idx;
        CudaDataDenseVector column_change(basis_size);
        double delta_z = 0;

        for (int i = 0; i < non_basis_indexes.getSize(); i++)
        {
            int j = non_basis_indexes[i];
            d[j] = d[j] - theta * alpha_p[i];  
            if (problem->bound_type[j] == BoundaryType::Boxed)
            {
                if (fabs(x[j] - problem->lower_bound[j]) < EPS_Z && d[j] < -EPS_D)
                {
                    low_infeas_idx.push_back(j);
                    delta_z += problem->costs[j] * (problem->upper_bound[j] - problem->lower_bound[j]);
                }  
                else if (fabs(x[j] - problem->upper_bound[j]) < EPS_Z && d[j] > EPS_D)
                {
                    up_infeas_idx.push_back(j);
                    delta_z -= problem->costs[j] * (problem->upper_bound[j] - problem->lower_bound[j]);
                }   
            }          
        }
        
        problem->A.addSparseCol(sp_handle, column_change, low_infeas_idx, difference_bounds, 1); 
        problem->A.addSparseCol(sp_handle, column_change, up_infeas_idx, difference_bounds, -1); 

        d[p] = -theta;
        d[q] = 0;
        _timer->stopTimer(ALgorithmPart::UpdateRedCosts);

        if (up_infeas_idx.size() || low_infeas_idx.size())
        {
            _timer->startTimer();
            CudaDataDenseVector delta_xB(basis_size);
            CudaDataDenseVector rhs(basis_size);

            solveLinSys(column_change, delta_xB, false);
            delta_xB.updateHostMem();

            rhs.updateByPartialVec(x, basis_indexes);
            rhs.updateDeviceMem();
            rhs.axpyUpdate(cu_handle, delta_xB, -1);
            cudaDeviceSynchronize();
            rhs.updateHostMem();
            x.setValues(rhs, basis_indexes);
           
            for (int i = 0; i < basis_size; i++)
            {
                int j = basis_indexes[i];
                delta_z -= problem->costs[j] * delta_xB[i];
            }    
            _timer->stopTimer(ALgorithmPart::FtranBfrt);
        }
        obj_func_val += delta_z;

        // Update B and xB, DSE weights
        _timer->startTimer();
        double theta_P = delta / alpha_q[p_idx];
        updateAndChangeBasis(x, rho, alpha_q, p_idx, p, q_idx, q, theta_P);
        _timer->stopTimer(ALgorithmPart::BasisUpate);

        // Flip bounds
        #pragma omp parallel for
        for (auto j : low_infeas_idx) x[j] = problem->upper_bound[j];
        #pragma omp parallel for
        for (auto j : up_infeas_idx)  x[j] = problem->lower_bound[j]; 
        
        obj_func_val += theta * delta;

        if (fabs(theta) < EPS_A) cycle_num += 1;

        #ifdef DEBUG
            std::cout << iteration << " : Z = "<< obj_func_val  << " delta_z = " 
                      << delta_z << " step_val = " << theta * delta  << " theta = " 
                      << theta << " delta = " << delta << " p_info:" << p << " q_info:" << q  
                      <<  " inf:" << counterDualInfeasible() << " beta[p_idx] = " << beta[p_idx] << std::endl;
        #endif
    }
    std::cout << "iterations = " << iteration << std::endl;
    _timer->printInfo();
    _timer->reset();
    std::cout << "cudss: " << ker << std::endl;
    std::cout << "pfi: " << pfi << std::endl;
    return problem->solution.solved;
}

