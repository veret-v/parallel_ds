#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <cmath>
#include <random>
#include <chrono>
#include <set>
#include <set>
#include <queue>
#include <algorithm>
#include <memory>
#include <mpi.h>

#include "linalg.hpp"
#include "valuesVector.hpp"
#include "matrix.hpp"

#include "../common/problem.hpp"
#include "../common/types.hpp"
#include "../common/LPsolution.hpp"
#include "../common/timeProfiler.hpp"
#include "../common/baseDualSimplex.hpp"


class ParallelDualSimplex : public BaseDualSimplex<Matrix, ValuesVector, IndexVector>
{
protected:
    std::vector<EtaMatrix> B_eta_repr;
    std::unique_ptr<TimeProfiler> _timer;

    int _receive_pricing_flag = 0;
    int _send_update_flag = 0;
    int _need_refact_flag = 0;
    int _solved_flag = 0;

    int empty_message_ = 0;

    std::vector<int> _workers;
    int _my_rank;
    int _my_master;

    int _in_work_size;

    int _max_cand_num;
    int _curr_cand_num;


    std::vector<Candidate> _candidates;
    std::vector<int> _cand_order;
    
    std::vector<double> _matrix_buff;
    std::vector<int> _id_matrix_buff;


    MPI_Status _curr_status;

    
    // for mpi
    void recvRefact();
    void sendRefact();

    void recvAndExecBfrtBeta();
    void sendTaskForBfrtBeta();

    void recvBtran();
    void sendBtran();

    void recvUpdateBasis();
    void sendStartUpdateBasis(); // send first part of eta matrix
    void sendFinishUpdateBasis(); // send last part of eta matrix
    void sendFullUpdateBasis();

    void minorUpate(int curr_cand, const IndexVector& low_infeas_idx, const IndexVector& up_infeas_idx);
    void majorUpdate();

    void updateRho(int curr_cand);
    void updateApf(int curr_cand);

    PresolverMethods stringToPreSolverMethod(const std::string& method_name) override;

    bool callPrimalSolver() override;

    void solveLinSys(
        ValuesVector&& rhs, 
        ValuesVector& sol,
        bool transpose
    ) override;

    void solveLinSys(
        ValuesVector& rhs, 
        ValuesVector& sol,
        bool transpose
    ) override;


    void parallelDSEPricing(std::unordered_set<int>& blocked_p);

    void BTran(int p_idx, ValuesVector& rho) override;
    void pivotRow(ValuesVector& rho, ValuesVector& alpha) override;
    void FTran(int q, ValuesVector& alpha_q) override;
    void initReducedCosts() override;
    void initPhase1PricingVector(ValuesVector& f, IndexVector& inf_u_indexes, IndexVector& inf_l_indexes, double& Z) override;
    void reFactorize() override;
    void simpleReducedCostsUpate(const ValuesVector& alpha, int q_idx, int q, double theta) override;
    void phase1UpdateAndChangeBasis(
        ValuesVector& f, ValuesVector& rho, const ValuesVector& alpha_q, 
        int p_idx, int p, int q_idx, int q, double theta_P
    ) override;

    void dualSimplexInit() override;

public:
    using BaseDualSimplex::BaseDualSimplex;

    void initMaster(int my_rank, int world_size);
    void initWorker(int master_rank, int my_rank);

    bool masterStage();
    bool workerStage();

    void solveMaster();
    void solveWorker();
};


