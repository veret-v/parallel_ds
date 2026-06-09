#include "sequentialDualSimplex.hpp"


void SequentialDualSimplex::initDualSimplex()
{
    auto size = problem->A.getSize();
    std::cout << "Problem size: " << std::get<0>(size) << "x" << std::get<1>(size) << std::endl;

    std::cout << "Solver initialization : basis columns selected" << std::endl;

    x = ValuesVector(full_size);
    d = ValuesVector(full_size);    

    problem->A.genSparseReprs();  

    B.resetData(problem->A, basis_indexes);
    B_eta_repr.reserve(REFACT_FREQ);
    B_eta_repr.clear();
    B.LUdecompose();
    
    _timer = std::make_unique<TimeProfiler>(TimeProfiler());


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


void SequentialDualSimplex::BTran(int p_idx, ValuesVector& rho) 
{
    solveLinSys(linalg::unit(basis_size, p_idx), rho, true);
}


void SequentialDualSimplex::pivotRow(ValuesVector& rho, ValuesVector& alpha) 
{
    problem->A.dotUpdate(
        rho, problem->costs, 
        alpha, 1, 0, 
        non_basis_indexes, 
        SpmvOptions::UPDATE_T,
        false
    );
}


void SequentialDualSimplex::FTran(int q, ValuesVector& alpha_q)
{
    solveLinSys(problem->A(q), alpha_q, false);
}


bool SequentialDualSimplex::callPrimalSolver()
{
   return true;
}


void SequentialDualSimplex::initReducedCosts()
{
    ValuesVector y(basis_size);

    solveLinSys(problem->costs(basis_indexes), y, true);
   
    problem->A.dotUpdate(
        y, problem->costs, 
        d, -1, 1, 
        non_basis_indexes, 
        SpmvOptions::UPDATE_T,
        true
    );
}


void SequentialDualSimplex::initPhase1PricingVector(ValuesVector& f, IndexVector& inf_u_indexes, IndexVector& inf_l_indexes, double& Z)
{
    ValuesVector columns_change(basis_size);  

    Z = 0;
    for (auto i : inf_l_indexes)
    {
        Z += d[i];
        columns_change += problem->A(i);
    }

    for (auto i : inf_u_indexes)
    {
        Z -= d[i];
        columns_change -= problem->A(i);
    }
    solveLinSys(columns_change, f, false);
}


void SequentialDualSimplex::reFactorize()
{
    B_eta_repr.clear();
    B.resetData(problem->A, basis_indexes);
    B.LUdecompose();
}


void SequentialDualSimplex::simpleReducedCostsUpate(const ValuesVector& alpha, int p, int q, double theta)
{
    for (int i = 0; i < non_basis_indexes.size(); i++)
    {
        int j = non_basis_indexes[i];
        d[j] -= theta * alpha[i];    
    }
    d[p] = -theta;
    d[q] = 0;
}

void SequentialDualSimplex::phase1UpdateAndChangeBasis(
    ValuesVector& f, ValuesVector& rho, const ValuesVector& alpha_q, 
    int p_idx, int p, int q_idx, int q, double theta_P
)
{
    ValuesVector new_eta_matrix(basis_size);        // basis_size
    ValuesVector tau(basis_size);                   // basis_size

    solveLinSys(rho, tau, false);

    for (int i = 0; i < basis_size; i++)
    {
        int j = basis_indexes[i];
        f[i] = (i != p_idx) ? f[i] - alpha_q[i] * theta_P : f[p_idx];
        new_eta_matrix[i] = (i != p_idx) ? -alpha_q[i] / alpha_q[p_idx] : 1 / alpha_q[p_idx]; 
        beta[i] = (i != p_idx) ? beta[i] - 2 * alpha_q[i] / alpha_q[p_idx] * tau[i]  + pow(alpha_q[i] / alpha_q[p_idx], 2) * beta[p_idx] : beta[i]; 
    }  
    beta[p_idx] = beta[p_idx] / pow(alpha_q[p_idx], 2);

    basis_indexes[p_idx] =  q;
    non_basis_indexes[q_idx] = p;

    B_eta_repr.push_back(EtaMatrix(new_eta_matrix, p_idx));
}


void SequentialDualSimplex::updateAndChangeBasis(
    ValuesVector& f, ValuesVector& rho, const ValuesVector& alpha_q, 
    int p_idx, int p, int q_idx, int q, double theta_P
)
{
    ValuesVector new_eta_matrix(basis_size);        // basis_size
    ValuesVector tau(basis_size);                   // basis_size

    _timer->startTimer();
    solveLinSys(rho, tau, false);
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
    basis_indexes[p_idx] =  q;
    non_basis_indexes[q_idx] = p;

    B_eta_repr.push_back(EtaMatrix(new_eta_matrix, p_idx));
}


void SequentialDualSimplex::dualSimplexInit()
{
    ValuesVector buff_sol(basis_size);
    ValuesVector rhs(basis_size);

    setPrimalVars();

    problem->A.dotUpdate(
        x, problem->RHS, 
        rhs, -1, 1, 
        non_basis_indexes, 
        SpmvOptions::UPDATE,
        true
    );

    solveLinSys(rhs, buff_sol, false);
    x.setValues(buff_sol, basis_indexes);

    initReducedCosts();
    obj_func_val = problem->costs.dot(x);
}


bool SequentialDualSimplex::callDualSolver()
{
    // (Step 1) Initialization
    ValuesVector rho(basis_size);
    ValuesVector alpha_p(non_basis_size);
    ValuesVector alpha_q(basis_size);
    ValuesVector buff_sol(basis_size);
    ValuesVector rhs(basis_size);

    ValuesVector difference_bounds = problem->upper_bound - problem->lower_bound;
        
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
        _timer->startTimer();
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
        _timer->stopTimer(ALgorithmPart::Pricing);
        
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
                    (problem->bound_type[j] == BoundaryType::Lower || 
                     problem->bound_type[j] == BoundaryType::Boxed))
                    cond = true;
            } else if (a < -EPS_A) {
                int j = non_basis_indexes[i];
                double xj = x[j];
                if (fabs(xj - problem->upper_bound[j]) < EPS_Z &&
                    (problem->bound_type[j] == BoundaryType::Upper || 
                     problem->bound_type[j] == BoundaryType::Boxed))
                    cond = true;
            } else {
                int j = non_basis_indexes[i];
                if (problem->bound_type[j] == BoundaryType::Free)
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
            unbound_checked = (unbound_checked) ? false : false;
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
        if (fabs(alpha_q[p_idx] - alpha_p[q_idx]) > EPS_R * (1 + fabs(alpha_q[p_idx])))
        {
            _timer->startTimer();
            reFactorize();
            _timer->stopTimer(ALgorithmPart::Factor);        
        }

        // (Step 7) Basis change and update
        // Update d accoprding to BRFT
        _timer->startTimer();
        IndexVector low_infeas_idx, up_infeas_idx;
        ValuesVector column_change(basis_size);
        double delta_z = 0;

        for (int i = 0; i < non_basis_indexes.size(); i++)
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
       
        problem->A.addSparseCol(column_change, low_infeas_idx, difference_bounds, 1); 
        problem->A.addSparseCol(column_change, up_infeas_idx, difference_bounds, -1); 

        d[p] = -theta;
        d[q] = 0;
        _timer->stopTimer(ALgorithmPart::UpdateRedCosts);

        if (up_infeas_idx.size() || low_infeas_idx.size())
        {
            _timer->startTimer();
            ValuesVector delta_xB(basis_size);
            solveLinSys(column_change, delta_xB, false);
            x.setValues(x(basis_indexes) - delta_xB, basis_indexes);

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
        for (auto j : low_infeas_idx) x[j] = problem->upper_bound[j];
        for (auto j : up_infeas_idx)  x[j] = problem->lower_bound[j]; 

        obj_func_val += theta * delta;

        #ifdef DEBUG
            std::cout << iteration << " : Z = "<< obj_func_val  << " delta_z = " << delta_z 
                      << " delta = " << theta * delta  << " theta = " << theta << " delta = " 
                      << delta << " p_info:" << p << " q_info:" << q  <<  " inf:" << counterDualInfeasible() 
                      << " beta[p_idx] = " << beta[p_idx] << std::endl;
        #else
            if (iteration % 500 == 0)
            {
                std::cout << iteration << " : Z = "<< obj_func_val  << std::endl;
            }
        #endif
        if (fabs(theta) < EPS_A) cycle_num += 1;
    }
    std::cout << "iterations = " << iteration << std::endl;
    _timer->printInfo();
    _timer->reset();

    return problem->solution.solved;
}

