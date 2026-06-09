#include "parallelDualSimplex.hpp"


void ParallelDualSimplex::initMaster(int my_rank, int world_size, double psi)
{
    _psi = psi;

    _my_rank = my_rank;
    _my_master = 0;
    _max_cand_num = world_size - 1;

    _workers.reserve(world_size);
    for (int i = 1; i < world_size; i++)
    {
        _workers.push_back(i);
    }

    
    auto size = problem->A.getSize();
    std::cout << "Problem size: " << std::get<0>(size) << "x" << std::get<1>(size) << std::endl;

    std::cout << "Solver initialization : basis columns selected" << std::endl;

    x = ValuesVector(full_size);
    d = ValuesVector(full_size);    

    problem->A.genSparseReprs();  

    B.resetData(problem->A, basis_indexes);
    B_eta_repr.clear();
    B_eta_repr.reserve(REFACT_FREQ + 10);
    B.LUdecompose();
    
    _timer = std::make_unique<TimeProfiler>(TimeProfiler());


    std::cout << "Solver initialization : attributes setted" << std::endl; 
}


 void ParallelDualSimplex::initWorker(int master_rank, int my_rank)
 {
    _my_rank = my_rank;
    _my_master = master_rank;

    x = ValuesVector(full_size);
    d = ValuesVector(full_size);    

    problem->A.genSparseReprs();  

    B.resetData(problem->A, basis_indexes);
    B_eta_repr.clear();
    B_eta_repr.reserve(REFACT_FREQ + 10);
    B.LUdecompose();
 }


void ParallelDualSimplex::solveLinSys(
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


void ParallelDualSimplex::solveLinSys(
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


void ParallelDualSimplex::BTran(int p_idx, ValuesVector& rho) 
{
    solveLinSys(linalg::unit(basis_size, p_idx), rho, true);
}


void ParallelDualSimplex::pivotRow(ValuesVector& rho, ValuesVector& alpha) 
{
    double sparsity = rho.genSparse();

    if (sparsity < 0.3)
    {
        problem->A.dotUpdate(
            rho, problem->costs, 
            alpha, 1, 0, 
            non_basis_indexes, 
            SpmvOptions::UPDATE_T_SP_ROW_WISE,
            false
        );
    }
    else
    {
        problem->A.dotUpdate(
            rho, problem->costs, 
            alpha, 1, 0, 
            non_basis_indexes, 
            SpmvOptions::UPDATE_T_SP_COL_WISE,
            false
        );
    }
}


void ParallelDualSimplex::FTran(int q, ValuesVector& alpha_q)
{
    solveLinSys(problem->A(q), alpha_q, false);
}


bool ParallelDualSimplex::callPrimalSolver()
{
   return true;
}


void ParallelDualSimplex::initReducedCosts()
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


void ParallelDualSimplex::initPhase1PricingVector(ValuesVector& f, IndexVector& inf_u_indexes, IndexVector& inf_l_indexes, double& Z)
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


void ParallelDualSimplex::reFactorize()
{
    B_eta_repr.clear();
    B.resetData(problem->A, basis_indexes);
    B.LUdecompose();
}


void ParallelDualSimplex::simpleReducedCostsUpate(const ValuesVector& alpha, int p, int q, double theta)
{
    #pragma omp parallel for
    for (int i = 0; i < non_basis_indexes.size(); i++)
    {
        int j = non_basis_indexes[i];
        d[j] -= theta * alpha[i];    
    }
    d[p] = -theta;
    d[q] = 0;
}

void ParallelDualSimplex::phase1UpdateAndChangeBasis(
    ValuesVector& f, ValuesVector& rho, const ValuesVector& alpha_q, 
    int p_idx, int p, int q_idx, int q, double theta_P
)
{
    ValuesVector new_eta_matrix(basis_size);        // basis_size
    ValuesVector tau(basis_size);                   // basis_size

    solveLinSys(rho, tau, false);

    #pragma omp parallel for
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

    B_eta_repr.emplace_back(EtaMatrix(new_eta_matrix, p_idx));
}


void ParallelDualSimplex::majorUpdate()
{
    // restore x, basis_indexes, non_basis_indexes
    auto start = _cand_order.rbegin();
    auto end   = _cand_order.rend();
    for(start; start != end; start += 1)
    {
        int p_idx = _candidates[*start].p_idx_;
        int p     = _candidates[*start].p_;
        int q_idx = _candidates[*start].q_idx_;
        int q     = _candidates[*start].q_;

        x[p] =  _candidates[*start].x_p_;
        basis_indexes[p_idx] =  p;
        non_basis_indexes[q_idx] = q;
    }

    double delta_z = 0;
    for (auto id : _cand_order)
    {
        if (_candidates[id].bfrt_done_)
        {
            ValuesVector& curr_delta_xB = _candidates[id].delta_xB_;
            x.setValues(x(basis_indexes) - curr_delta_xB, basis_indexes);
            #pragma omp parallel for 
            for (int i = 0; i < basis_size; ++i)
            {
                int j = basis_indexes[i];
                delta_z -= problem->costs[j] * curr_delta_xB[i];
            }
        }
    
        ValuesVector& tau     = _candidates[id].tau_;
        ValuesVector& alpha_q = _candidates[id].alpha_q_;
        int p_idx             = _candidates[id].p_idx_;
        int p                 = _candidates[id].p_;
        int q_idx             = _candidates[id].q_idx_;
        int q                 = _candidates[id].q_;
        double theta_P        = _candidates[id].delta_ / alpha_q[p_idx];

        #pragma omp parallel for 
        for (int i = 0; i < basis_size; i++)
        {
            int j = basis_indexes[i];
            x[j] = x[j] - theta_P * alpha_q[i];  
            beta[i] = (i != p_idx) ? beta[i] - 2 * alpha_q[i] / alpha_q[p_idx] * tau[i]  + pow(alpha_q[i] / alpha_q[p_idx], 2) * beta[p_idx] : beta[i]; 
        }  
        beta[p_idx] = beta[p_idx] / pow(alpha_q[p_idx], 2);
        x[q] = x[q] + theta_P;

        _mask_x_at_lower[q_idx] = (fabs(x[p] - problem->lower_bound[p]) < EPS_Z);
        _mask_x_at_upper[q_idx] = (fabs(x[p] - problem->upper_bound[p]) < EPS_Z);

        basis_indexes[p_idx] =  q;
        non_basis_indexes[q_idx] = p;
    }

    obj_func_val += delta_z;
}


void ParallelDualSimplex::minorUpate(int curr_cand, const IndexVector& low_infeas_idx, const IndexVector& up_infeas_idx)
{
    int p_idx             = _candidates[curr_cand].p_idx_;
    ValuesVector& alpha_q = _candidates[curr_cand].alpha_q_;
    double theta_P        = _candidates[curr_cand].delta_ / alpha_q[p_idx];
    int q                 = _candidates[curr_cand].q_;
    int q_idx             = _candidates[curr_cand].q_idx_;
    int p                 = _candidates[curr_cand].p_;
    int is_lower          = _candidates[curr_cand].is_lower_;

    ValuesVector new_eta_matrix(basis_size);   
    #pragma omp parallel for
    for (int i = 0; i < basis_size; i++)
        new_eta_matrix[i] = (i != p_idx) ? -alpha_q[i] / alpha_q[p_idx] : 1 / alpha_q[p_idx]; 
    
    // set non basis p, need for ratio  test
    _candidates[curr_cand].x_p_ = x[p];
    x[p] = (is_lower) ? problem->lower_bound[p] : problem->upper_bound[p];

    // update deltas acording bfrt and classic
    #pragma omp parallel for 
    for (int i = 0; i < _curr_cand_num; ++i)
    {
        if (_candidates[i].is_active_)
        {
            for (auto j : low_infeas_idx)
                _candidates[i].delta_ -= (problem->upper_bound[j] - problem->lower_bound[j]) * problem->A.dotCol(_candidates[i].rho_, j);
            
            for (auto j : up_infeas_idx)
                _candidates[i].delta_ += (problem->upper_bound[j] - problem->lower_bound[j]) * problem->A.dotCol(_candidates[i].rho_, j);
        
            _candidates[i].delta_ -= theta_P * alpha_q[ _candidates[i].p_idx_];
        }
    }
    

    if (problem->bound_type[q] == BoundaryType::Boxed) 
        _boxed_in_non_basis.erase(std::remove(_boxed_in_non_basis.begin(), _boxed_in_non_basis.end(), q_idx), _boxed_in_non_basis.end());
    if (problem->bound_type[p] == BoundaryType::Boxed) 
        _boxed_in_non_basis.push_back(q_idx);

    basis_indexes[p_idx] =  q;
    non_basis_indexes[q_idx] = p;

    _matrix_buff.insert(_matrix_buff.end(), new_eta_matrix.begin(), new_eta_matrix.end());
    _id_matrix_buff.push_back(p_idx);
    B_eta_repr.push_back(EtaMatrix(new_eta_matrix, p_idx));
}


void ParallelDualSimplex::updateApf(int curr_cand)
{
    problem->A.addSparseCol(_candidates[curr_cand].APF_update_, _candidates[curr_cand].q_, 1); 
    problem->A.addSparseCol(_candidates[curr_cand].APF_update_, _candidates[curr_cand].p_, -1);
    double mu = 1 + cblas_ddot(basis_size, _candidates[curr_cand].APF_update_.getPointerData(), 1, _candidates[curr_cand].rho_.getPointerData(), 1);
    _candidates[curr_cand].APF_update_ = _candidates[curr_cand].APF_update_ * (1 / mu);
}


void ParallelDualSimplex::updateRho(int curr_cand)
{
    
    for (int i = 0; i < _curr_cand_num; ++i)
    {
        if (_candidates[i].is_active_)
        {
            linalg::APFsolve(
                _candidates[curr_cand].APF_update_, _candidates[curr_cand].rho_, 
                _candidates[i].rho_, true
            );
        }
    }

   
}


void ParallelDualSimplex::dualSimplexInit()
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

void ParallelDualSimplex::solveMaster()
{
    std::cout << "Phase 2 : started" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    bool status_code = masterStage();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
   
    std::cout << "time = " << duration_ms / 1000 << " s" << std::endl;

    postProcess(status_code);

    std::cout << "Phase 2 : ended" << std::endl;
}


void ParallelDualSimplex::solveWorker()
{
    bool status_code = workerStage();
}

void ParallelDualSimplex::parallelDSEPricing(std::unordered_set<int>& blocked_p)
{
    double delta;
    int p = -1;
    int p_idx = -1;
    bool is_lower = false;
    double max_weight = 0;

    std::vector<double> weights_cand;
    for (int i = 0; i < basis_size; i++)
    {
        int j = basis_indexes[i];
        auto [is_infeas, delta_tmp] = checkVarIsInfeas(j);
        bool is_lower_tmp; 

        if (blocked_p.find(j) == blocked_p.end() && is_infeas && fabs(delta_tmp) > 1e-10)
        {
            double weight_tmp = pow(delta_tmp, 2) / beta[i];
            is_lower_tmp = delta_tmp < 0;

            if (weight_tmp > max_weight)
            {
                if (_candidates.size() < _max_cand_num)
                {
                    _candidates.emplace_back(
                        Candidate(
                            i, j, 0, 0, 
                            is_lower_tmp, false, true, 
                            delta_tmp, weight_tmp, 
                            ValuesVector(basis_size), ValuesVector(basis_size), 
                            ValuesVector(basis_size),  ValuesVector(basis_size), 
                            ValuesVector(basis_size), ValuesVector(basis_size)
                        )
                    );
                    weights_cand.push_back(weight_tmp);
                }
                else
                {
                    auto min_it = std::min_element(weights_cand.begin(), weights_cand.end());
                    int index = std::distance(weights_cand.begin(), min_it);
                    if (weights_cand[index] < weight_tmp)
                    {
                        _candidates[index].p_idx_    = i;
                        _candidates[index].p_        = j;
                        _candidates[index].is_lower_ = is_lower_tmp;
                        _candidates[index].delta_    = delta_tmp;
                        _candidates[index].weight_   = weight_tmp;
                        
                        weights_cand[index] = weight_tmp;
                    }
                }
            }
        }
    }
}


void ParallelDualSimplex::recvRefact()
{
    MPI_Recv(non_basis_indexes.data(), non_basis_size, MPI_INT, _my_master, MpiTag::Refact, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(basis_indexes.data(), basis_size, MPI_INT, _my_master, MpiTag::Refact, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  

    reFactorize();
}


void ParallelDualSimplex::sendRefact()
{
    for (auto worker_id : _workers)
    {
        MPI_Send(non_basis_indexes.data(), non_basis_size ,MPI_INT, worker_id, MpiTag::Refact, MPI_COMM_WORLD);
        MPI_Send(basis_indexes.data(), basis_size, MPI_INT, worker_id, MpiTag::Refact, MPI_COMM_WORLD);
    }
}


void ParallelDualSimplex::recvUpdateBasis()
{
    ValuesVector new_eta_matrix(basis_size);

    MPI_Get_count(&_curr_status, MPI_INT, &_curr_cand_num);

    _id_matrix_buff.resize(_curr_cand_num);
    _matrix_buff.resize(basis_size * _curr_cand_num);

    MPI_Recv(_id_matrix_buff.data(), _curr_cand_num, MPI_INT, _my_master, MpiTag::TransferBasisChange, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(_matrix_buff.data(), basis_size * _curr_cand_num, MPI_DOUBLE, _my_master, MpiTag::TransferBasisChange, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    if (_curr_cand_num)
    {
        int i = 0;
        for (auto p_idx : _id_matrix_buff)
        {
            std::copy(_matrix_buff.begin() + basis_size * i, _matrix_buff.begin() + basis_size * (i + 1), new_eta_matrix.begin());
            B_eta_repr.emplace_back(EtaMatrix(new_eta_matrix, p_idx));
            i++;
        }
    }

    _id_matrix_buff.clear();
    _matrix_buff.clear();
}
    

void ParallelDualSimplex::sendFullUpdateBasis()
{
    for (auto worker_id : _workers)
    {
        if (std::find(_cand_order.begin(), _cand_order.end(), worker_id - 1) == _cand_order.end())
        {
            MPI_Send(_id_matrix_buff.data(), _cand_order.size(), MPI_INT, worker_id, MpiTag::TransferBasisChange, MPI_COMM_WORLD);
            MPI_Send(_matrix_buff.data(), _cand_order.size() * basis_size, MPI_DOUBLE, worker_id, MpiTag::TransferBasisChange, MPI_COMM_WORLD);
            // std::cout <<  worker_id << " all " << _id_matrix_buff.size() << std::endl;
        }
    }
}


void ParallelDualSimplex::sendStartUpdateBasis()
{
    for (int id = 0; id < _cand_order.size(); id++)
    {
        MPI_Send(_id_matrix_buff.data(), id, MPI_INT, _workers[_cand_order[id]], MpiTag::TransferBasisChange, MPI_COMM_WORLD);
        MPI_Send(_matrix_buff.data(), id * basis_size, MPI_DOUBLE, _workers[_cand_order[id]], MpiTag::TransferBasisChange, MPI_COMM_WORLD);
        // std::cout <<  _workers[_cand_order[id]] << " start "  << id << std::endl;
    }
}


void ParallelDualSimplex::sendFinishUpdateBasis()
{
    int buff_size = _cand_order.size();
    for (int id = 0; id < _cand_order.size(); id++)
    {
        MPI_Send(_id_matrix_buff.data() + id, buff_size - id, MPI_INT, _workers[_cand_order[id]], MpiTag::TransferBasisChange, MPI_COMM_WORLD);
        MPI_Send(_matrix_buff.data() + id * basis_size, (buff_size - id) * basis_size, MPI_DOUBLE, _workers[_cand_order[id]], MpiTag::TransferBasisChange, MPI_COMM_WORLD);
        // std::cout <<  _workers[_cand_order[id]] << " finish "  << buff_size - id << std::endl;
    }
}


void ParallelDualSimplex::recvAndExecBfrtBeta()
{
    ValuesVector delta_xB(basis_size);
    ValuesVector tau(basis_size);
    ValuesVector rho(basis_size);
    ValuesVector column_change(basis_size);

    MPI_Recv(column_change.getPointerData(), basis_size, MPI_DOUBLE, _my_master, MpiTag::FTran_BFRT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(rho.getPointerData(), basis_size, MPI_DOUBLE, _my_master, MpiTag::FTran_BFRT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    solveLinSys(column_change, delta_xB, false);
    solveLinSys(rho, tau, false);

    MPI_Send(delta_xB.getPointerData(), basis_size, MPI_DOUBLE, _my_master, MpiTag::FTran_BFRT, MPI_COMM_WORLD);
    MPI_Send(tau.getPointerData(), basis_size, MPI_DOUBLE, _my_master, MpiTag::FTran_BFRT, MPI_COMM_WORLD);
}


void ParallelDualSimplex::sendTaskForBfrtBeta()
{
    for (auto cand_id : _cand_order)
    {
        MPI_Send(_candidates[cand_id].column_change_.getPointerData(), basis_size, MPI_DOUBLE, _workers[cand_id], MpiTag::FTran_BFRT, MPI_COMM_WORLD);
        MPI_Send(_candidates[cand_id].rho_.getPointerData(), basis_size, MPI_DOUBLE, _workers[cand_id], MpiTag::FTran_BFRT, MPI_COMM_WORLD);
    }
    

    for (auto cand_id : _cand_order)
    {
        MPI_Recv(_candidates[cand_id].delta_xB_.getPointerData(), basis_size, MPI_DOUBLE, _workers[cand_id], MpiTag::FTran_BFRT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(_candidates[cand_id].tau_.getPointerData(), basis_size, MPI_DOUBLE, _workers[cand_id], MpiTag::FTran_BFRT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}


void ParallelDualSimplex::recvBtran()
{
    ValuesVector rho(basis_size);
    int p_idx;
    MPI_Recv(&p_idx, 1, MPI_INT, _my_master, MpiTag::BTran, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    BTran(p_idx, rho);
    MPI_Send(rho.getPointerData(), basis_size, MPI_DOUBLE, _my_master, MpiTag::BTran, MPI_COMM_WORLD);
}


void ParallelDualSimplex::sendBtran()
{
    for (int i = 0; i < _curr_cand_num; ++i)
    {
        int worker = _workers[i];
        MPI_Send(&_candidates[i].p_idx_, 1, MPI_INT, worker, MpiTag::BTran, MPI_COMM_WORLD);
    }

    for (int i = 0; i < _curr_cand_num; ++i)
    {
        int worker = _workers[i];
        MPI_Recv(_candidates[i].rho_.getPointerData(), basis_size, MPI_DOUBLE, worker, MpiTag::BTran, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}


bool ParallelDualSimplex::workerStage()
{    
    _matrix_buff.reserve(basis_size * _max_cand_num);
    _id_matrix_buff.reserve(_max_cand_num);

    bool finished = false;
    while (!finished)
    {
        int tag;
        MPI_Probe(_my_master, MPI_ANY_TAG, MPI_COMM_WORLD, &_curr_status);
        tag = _curr_status.MPI_TAG;

        switch (tag)
        {
            case MpiTag::SolutionStatus:
                MPI_Recv(&_solved_flag, 1, MPI_INT, _my_master, MpiTag::SolutionStatus, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                finished = true;
                break;

            case MpiTag::BTran:
                recvBtran();
                break;

            case MpiTag::FTran_BFRT:
                recvAndExecBfrtBeta();
                break;

            case MpiTag::TransferBasisChange:
                recvUpdateBasis();
                break;

            case MpiTag::Refact:
                recvRefact();
                break;

            default:
                int dummy;
                MPI_Recv(&dummy, 1, MPI_INT, _my_master, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
        }
    }
    return problem->solution.solved;
}


void ParallelDualSimplex::initTestsMasks()
{
    _mask_x_at_lower.clear();
    _mask_x_at_upper.clear();;
    _mask_is_lower_or_boxed.clear();;
    _mask_is_upper_or_boxed.clear();;
    _mask_is_free.clear();;

    _mask_x_at_lower.resize(non_basis_size, false);
    _mask_x_at_upper.resize(non_basis_size, false);
    _mask_is_lower_or_boxed.resize(non_basis_size, false);
    _mask_is_upper_or_boxed.resize(non_basis_size, false);
    _mask_is_free.resize(non_basis_size, false);
   
    #pragma omp parallel for 
    for (int i = 0; i < non_basis_size; i++)
    {
        int j = non_basis_indexes[i];
        double xj = x[j];
        _mask_x_at_lower[i] = (fabs(xj - problem->lower_bound[j]) < EPS_Z);
        _mask_x_at_upper[i] = (fabs(xj - problem->upper_bound[j]) < EPS_Z);
        _mask_is_lower_or_boxed[i] = (problem->bound_type[j] == BoundaryType::Lower || problem->bound_type[j] == BoundaryType::Boxed);
        _mask_is_upper_or_boxed[i] = (problem->bound_type[j] == BoundaryType::Upper || problem->bound_type[j] == BoundaryType::Boxed);
        _mask_is_free[i] = (problem->bound_type[j] == BoundaryType::Free);
    }
    
    _boxed_in_non_basis.clear();
    _boxed_in_non_basis.reserve(non_basis_size);
    for (int i = 0; i < non_basis_size; i++)
    {
        int j = non_basis_indexes[i];
        if (problem->bound_type[j] == BoundaryType::Boxed) _boxed_in_non_basis.push_back(i);
    }
    
}


void ParallelDualSimplex::parallelRatioTestPart1(IndexVector& F, const ValuesVector& alpha_p)
{
    #pragma omp parallel
    {
        int cnt = 0;
        IndexVector local_F; local_F.reserve(non_basis_size / 8);
        #pragma omp for
        for (int i = 0; i < non_basis_size; ++i) {
            bool cond = false;
            double a = alpha_p[i];
            if (a > EPS_A) {
                if (_mask_x_at_lower[i] && (_mask_is_lower_or_boxed[i]))
                    cond = true;
            } else if (a < -EPS_A) {
                if (_mask_x_at_upper[i] && (_mask_is_upper_or_boxed[i]))
                    cond = true;
            } else if (a == 0) { // учтём Free
                if (_mask_is_free[i])
                    cond = true;
            }
            if (cond) 
            {
                local_F.push_back(i);
            }
        }

        #pragma omp critical
        F.insert(F.end(), local_F.begin(), local_F.end());
    }
}


bool ParallelDualSimplex::masterStage()
{    
    ValuesVector rho(basis_size), alpha_q(basis_size);
    ValuesVector column_change(basis_size);
    IndexVector low_infeas_idx, up_infeas_idx;

    _candidates.reserve(_max_cand_num);
    _cand_order.reserve(_max_cand_num);

    ValuesVector diff_bounds = problem->upper_bound - problem->lower_bound;

    _timer->startTimer();
    reFactorize();
    dualSimplexInit();
    sendRefact();
    initTestsMasks();
    _timer->stopTimer(ALgorithmPart::Init);

    int cycle_num = 0;
    int sum_mini_iter = 0;
    int major_iteration = 0;
    bool update_weights = false;
    std::unordered_set<int> blocked_p;
    
    bool unbound_checked = false;

    std::cout << iteration << " : Z = "<< obj_func_val  <<  " inf:" << counterDualInfeasible() << std::endl;

    while (true)
    {
        major_iteration++;

        _candidates.clear();
        _cand_order.clear();
        _id_matrix_buff.clear();
        _matrix_buff.clear();
        
        // ---------- Рестарты и рефакторизация ----------
        if (iteration % NEED_RESTART == 0 && !checkDualFeasible())
        {
            std::cout << "-- Soft restart. Dual infeasiblity found." << std::endl;
            _timer->startTimer();
            reFactorize();
            minimizeDualInfeasibility();
            dualSimplexInit();
            sendRefact();
            initTestsMasks();
            _timer->stopTimer(ALgorithmPart::RestoreProc);
        }
        if (!perturbed && cycle_num > MAX_CYCLE)
        {
            std::cout << "-- Soft restart. Perturb costs." << std::endl;
            _timer->startTimer();
            perturbCosts();
            reFactorize();
            dualSimplexInit();
            sendRefact();
            initTestsMasks();
            _timer->stopTimer(ALgorithmPart::RestoreProc);
        }
        if (iteration % REFACT_FREQ == 0)
        {
            _timer->startTimer();
            reFactorize();
            sendRefact();
            _timer->stopTimer(ALgorithmPart::Factor);
        }
        if (checkPrimalFeasible())
        {
            _solved_flag = 1;
            for (auto w : _workers)
                MPI_Send(&_solved_flag, 1, MPI_INT, w, MpiTag::SolutionStatus, MPI_COMM_WORLD);
            problem->solution.solved = true;
            problem->solution.message = "optimal solution";
            break;
        }

        // ---------- Pricing ----------
        _timer->startTimer();
        parallelDSEPricing(blocked_p);
        _curr_cand_num = std::min(static_cast<int>(_candidates.size()), _max_cand_num);
        _timer->stopTimer(ALgorithmPart::Pricing);
        if (_curr_cand_num == 0)
        {
            std::cout << "-- Soft restart. No p find." << std::endl;  
            _timer->startTimer();
            reFactorize();
            initBetaWeights(false);
            minimizeDualInfeasibility();
            dualSimplexInit();
            sendRefact();
            initTestsMasks();
            blocked_p.clear();
            _timer->stopTimer(ALgorithmPart::RestoreProc);
            continue;
        }
        

        // ---------- Minor initialisation ---------
        // send andd collect BTran 
        _timer->startTimer();
        sendBtran();
        _timer->stopTimer(ALgorithmPart::Btran);

        // ---------- Minor iterations ----------
        int best_cand = -1;
        int mini_iter = 0;
        while (!std::all_of(_candidates.begin(), _candidates.end(), [](Candidate v){return !v.is_active_;}))
        {
            iteration++;
            mini_iter++;

            _timer->startTimer();
            best_cand = -1;
            double best_weight = -1.0;
            for (int i = 0; i < _curr_cand_num; ++i)
            {
                if (!_candidates[i].is_active_) continue;

                double delta = _candidates[i].delta_;
                // is_infeas if sign of delta hasn't changed
                bool is_infeas = (_candidates[i].is_lower_ && (delta < 0)) || (!_candidates[i].is_lower_ && !(delta < 0));

                if (!is_infeas)
                {
                    _candidates[i].is_active_ = false;
                    continue;
                }

                double old_weight = _candidates[i].weight_;
                _candidates[i].weight_   = pow(delta, 2) /  beta[_candidates[i].p_idx_];

                if (_candidates[i].weight_ < _psi * old_weight)
                {
                    _candidates[i].is_active_ = false;
                    continue;
                }

                if (_candidates[i].weight_ > best_weight)
                {
                    best_weight = _candidates[i].weight_;
                    best_cand = i;
                }
            }

            if (best_cand == -1) 
                break;
            else 
            {
                _cand_order.push_back(best_cand);   
            }
            _timer->stopTimer(ALgorithmPart::Pricing);
            
            // Выбран кандидат best_i
            int    p_idx        = _candidates[best_cand].p_idx_;
            int    p            = _candidates[best_cand].p_;
            double delta        = _candidates[best_cand].delta_;
            bool   is_lower     = _candidates[best_cand].is_lower_;
            ValuesVector& rho_p = _candidates[best_cand].rho_;

            // Pivot row (spmv)
            _timer->startTimer();
            ValuesVector alpha_p(non_basis_size);
            pivotRow(rho_p, alpha_p);
             _timer->stopTimer(ALgorithmPart::PivotRow);

            _timer->startTimer();
            delta = fabs(delta);
            int sgn = (is_lower) ? -1 : 1;
            if (is_lower) alpha_p.multiplyData(-1);
            // Ratio test с учётом bound flips
            IndexVector F; F.reserve(non_basis_size);  
            parallelRatioTestPart1(F, alpha_p);
            _timer->stopTimer(ALgorithmPart::RatioTestPart1);

            // check that problem is really uboundedd, not because of numeric error
            if (!F.size())
            {
                if (unbound_checked)
                {
                    _solved_flag = 1;
                    for (auto w : _workers)
                        MPI_Send(&_solved_flag, 1, MPI_INT, w, MpiTag::SolutionStatus, MPI_COMM_WORLD);

                    problem->solution.solved = false;
                    problem->solution.message = "dual unbounded";
                    return problem->solution.solved;
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
                    sendRefact();
                    initTestsMasks();
                    _timer->stopTimer(ALgorithmPart::RestoreProc);

                    break;
                }
            }
            else
            {
                unbound_checked = (unbound_checked) ? false : false;
            }

            _timer->startTimer();
            double theta = INF;
            int q_idx, q;
            while (F.size())
            {
                std::tie(q, q_idx) = simpleRatioTest(F, alpha_p);
                if (problem->bound_type[q] == BoundaryType::Boxed)
                {
                    if (delta - diff_bounds[q] * fabs(alpha_p[q_idx]) <= 0)
                    {
                        break;
                    }
                    delta -= diff_bounds[q] * fabs(alpha_p[q_idx]);
                    F.erase(std::remove(F.begin(), F.end(), q_idx), F.end());
                }
                else
                {
                    break;
                }
            }
            
            if (is_lower) alpha_p.multiplyData(-1); 
            theta       = d[q] / alpha_p[q_idx];
            delta       = sgn * delta;

            _candidates[best_cand].q_     = q;
            _candidates[best_cand].q_idx_ = q_idx;
            _timer->stopTimer(ALgorithmPart::RatioTestPart2);

            _timer->startTimer();
            FTran(q, _candidates[best_cand].alpha_q_);
            _timer->stopTimer(ALgorithmPart::Ftran);
            if (fabs(alpha_q[p_idx] - alpha_p[q_idx]) > EPS_R * (1 + fabs(alpha_q[p_idx])))
            {
                _timer->startTimer();
                reFactorize();
                _timer->stopTimer(ALgorithmPart::Factor);        
            }

            // Обработка BFRT (bound flips) and d
             _timer->startTimer();
            IndexVector low_infeas_idx, up_infeas_idx;
            IndexVector swap_low_infeas_idx, swap_up_infeas_idx;
            low_infeas_idx.reserve(non_basis_size);
            up_infeas_idx.reserve(non_basis_size);
            swap_low_infeas_idx.reserve(non_basis_size);
            swap_up_infeas_idx.reserve(non_basis_size);
            double delta_z = 0.0;

             
            #pragma omp parallel for 
            for (int i = 0; i < non_basis_indexes.size(); i++)
            {
                int j = non_basis_indexes[i];
                d[j] = d[j] - theta * alpha_p[i];  
            }

            for (auto i : _boxed_in_non_basis)
            {
                if (i == q_idx) continue;

                int j = non_basis_indexes[i];
                if (_mask_x_at_lower[i] && d[j] < -EPS_D)
                {
                    low_infeas_idx.push_back(j);
                    swap_low_infeas_idx.push_back(i);
                    delta_z += problem->costs[j] * diff_bounds[j];
                    
                }  
                else if (_mask_x_at_upper[i] && d[j] > EPS_D)
                {
                    up_infeas_idx.push_back(j);
                    swap_up_infeas_idx.push_back(i);
                    delta_z -= problem->costs[j] * diff_bounds[j];
                }   
            }
            
            problem->A.addSparseColParallel(_candidates[best_cand].column_change_, low_infeas_idx, diff_bounds, 1);
            problem->A.addSparseColParallel(_candidates[best_cand].column_change_, up_infeas_idx, diff_bounds, -1);

            d[p] = -theta;
            d[q] = 0;
            _timer->stopTimer(ALgorithmPart::UpdateRedCosts);

            _candidates[best_cand].is_active_ = false;
            _candidates[best_cand].delta_ = delta;
            if (!low_infeas_idx.empty() || !up_infeas_idx.empty())
                _candidates[best_cand].bfrt_done_ = true;

            // Применяем bound flips к x
            // Обновление базиса и primal/dual переменных
            _timer->startTimer();
            #pragma omp parallel for 
            for (auto i : swap_low_infeas_idx) 
            {
                int j = non_basis_indexes[i];
                x[j] = problem->upper_bound[j];
                _mask_x_at_lower[i] = false;
                _mask_x_at_upper[i] = true;
            }
            #pragma omp parallel for 
            for (auto i : swap_up_infeas_idx)  
            {
                int j = non_basis_indexes[i];
                x[j] = problem->lower_bound[j];
                _mask_x_at_lower[i] = true;
                _mask_x_at_upper[i] = false;
            }

            // update masks
            _mask_is_lower_or_boxed[q_idx] = (problem->bound_type[p] == BoundaryType::Lower || problem->bound_type[p] == BoundaryType::Boxed);
            _mask_is_upper_or_boxed[q_idx] = (problem->bound_type[p] == BoundaryType::Upper || problem->bound_type[p] == BoundaryType::Boxed);
            _mask_is_free[q_idx] = (problem->bound_type[p] == BoundaryType::Free);
            _mask_x_at_lower[q_idx] = is_lower;
            _mask_x_at_upper[q_idx] = !is_lower;

            
            minorUpate(best_cand, low_infeas_idx, up_infeas_idx);
            updateApf(best_cand);
            updateRho(best_cand); // update rho for all active candidates
            _timer->stopTimer(ALgorithmPart::BasisUpate);
            // exit(0);

            obj_func_val += theta * delta + delta_z;
            cycle_num = (fabs(theta) < EPS_A) ? cycle_num + 1 : 0;
            
            #ifdef DEBUG
                std::cout << "\t" << mini_iter << " : Z = "<< obj_func_val << " step_delta = " 
                          << theta * delta << " p_info:" << p << " q_info:" << q  <<  " inf:" 
                          << counterDualInfeasible() << " bfrt_done:" << _candidates[best_cand].bfrt_done_ << std::endl;
            #endif
        } // конец minor итераций

        if (unbound_checked) continue;
        
        sendStartUpdateBasis();
         _timer->startTimer();
        sendTaskForBfrtBeta();
        _timer->stopTimer(ALgorithmPart::FtranBfrt);

        _timer->startTimer();
        majorUpdate();
        // initTestsMasks();
        _timer->stopTimer(ALgorithmPart::BasisUpate);

        sendFullUpdateBasis(); // for workers not working in that iteration
        sendFinishUpdateBasis();

        #ifdef DEBUG
            std::cout << iteration << " : Z = "<< obj_func_val <<  " dual inf:" << counterDualInfeasible() <<  " primal inf:" << counterPrimalInfeasible() << std::endl;
        #else
            if (iteration % 500 == 0)
                std::cout << major_iteration << " : Z = "
                << obj_func_val  
                << " aver_mini_iter_num = " 
                << (double) sum_mini_iter / (double) major_iteration << std::endl;
        #endif
        sum_mini_iter += mini_iter;
    }

    std::cout << "major iterations = " << major_iteration << std::endl;
    std::cout << "average mini iter num = " << (double) sum_mini_iter / (double) major_iteration << std::endl;

    _timer->printInfo();
    _timer->reset();

    return problem->solution.solved;
}

