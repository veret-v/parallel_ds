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
#include <queue>
#include <algorithm>

#include "cudaDataDenseVector.hpp"
#include "cudaIndexVector.hpp"
#include "cudaSparseMatrix.hpp"
#include "PFIfactor.hpp"

#include "../common/problem.hpp"
#include "../common/types.hpp"
#include "../common/LPsolution.hpp"
#include "../common/timeProfiler.hpp"
#include "../common/baseDualSimplex.hpp"


class CudaDualSimplex : public BaseDualSimplex<CudaSparseMatrix, CudaDataDenseVector, CudaIndexVector>
{
protected:
    PFIfactor pfi_factor;

    cudaStream_t stream     = NULL;
    
    cublasHandle_t cu_handle   = NULL;
    cusparseHandle_t sp_handle = NULL;

    cudssHandle_t cudss_handle = NULL;
    cudssConfig_t cudss_config = NULL;
    cudssData_t cudss_data     = NULL;

    cudssHandle_t cudss_handle_T = NULL;
    cudssConfig_t cudss_config_T = NULL;
    cudssData_t cudss_data_T     = NULL;

    double pfi = 0;
    double ker =0;

    std::unique_ptr<TimeProfiler> _timer;

    void solveLinSys(
        CudaDataDenseVector& rhs, 
        CudaDataDenseVector& sol,
        bool transpose
    ) override;

    bool callDualSolver() override;
    bool callPrimalSolver() override;

    PresolverMethods stringToPreSolverMethod(const std::string& method_name) override;

    void BTran(int p_idx, CudaDataDenseVector& rho) override;
    void pivotRow(CudaDataDenseVector& rho, CudaDataDenseVector& alpha) override;
    void FTran(int q, CudaDataDenseVector& alpha_q) override;
    void initReducedCosts() override;
    void initPhase1PricingVector(CudaDataDenseVector& f, IndexVector& inf_u_indexes, IndexVector& inf_l_indexes, double& Z) override;
    void reFactorize() override;
    void simpleReducedCostsUpate(const CudaDataDenseVector& alpha, int p, int q, double theta) override;
    void updateAndChangeBasis(
        CudaDataDenseVector& f, CudaDataDenseVector& rho, const CudaDataDenseVector& alpha_q, 
        int p_idx, int p, int q_idx, int q, double theta_P
    ) override;
    void phase1UpdateAndChangeBasis(
        CudaDataDenseVector& f, CudaDataDenseVector& rho, const CudaDataDenseVector& alpha_q, 
        int p_idx, int p, int q_idx, int q, double theta_P
    ) override;

    void dualSimplexInit() override;

public:
    CudaDualSimplex(Problem<CudaSparseMatrix, CudaDataDenseVector>& _problem) : 
        BaseDualSimplex<CudaSparseMatrix, CudaDataDenseVector, CudaIndexVector>(_problem),
        pfi_factor(_problem.constraints_size, REFACT_FREQ + 5)
    {};
    using BaseDualSimplex::BaseDualSimplex;
    ~CudaDualSimplex();

    // Init params, setting data according to index arrays
    void initDualSimplex();
};


