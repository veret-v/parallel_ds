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
#include "../common/baseDualSimplex.hpp"


class CudaDualSimplex : public BaseDualSimplex<CudaSparseMatrix, CudaDataDenseVector, CudaIndexVector>
{
protected:
    PFIfactor pfi_factor;

    CudaDataDenseVector beta;

    cudaStream_t stream     = NULL;
    
    cublasHandle_t cu_handle   = NULL;
    cusparseHandle_t sp_handle = NULL;

    cudssHandle_t cudss_handle = NULL;
    cudssConfig_t cudss_config = NULL;
    cudssData_t cudss_data     = NULL;

    cudssHandle_t cudss_handle_T = NULL;
    cudssConfig_t cudss_config_T = NULL;
    cudssData_t cudss_data_T     = NULL;

    void solveLinSys(const bool transpose, const CudaDataDenseVector& rhs, CudaDataDenseVector& sol);

    Phase1OutStatus callPresolver(const PresolverMethods method) override;
    bool callDualSolver(const SolverMethods method) override;
    bool callPrimalSolver() override;
    
    SolverMethods    stringToSolverMethod(const std::string& method_name) override;
    PresolverMethods stringToPreSolverMethod(const std::string& method_name) override;

    Phase1OutStatus minimizeDualInfeasibility();

    bool elaboratedMethod();

    void initBetaWeights();

public:
    CudaDualSimplex(Problem<CudaSparseMatrix, CudaDataDenseVector>& _problem) : 
        BaseDualSimplex<CudaSparseMatrix, CudaDataDenseVector, CudaIndexVector>(_problem),
        pfi_factor(_problem.constraints_size, REFACT_FREQ + 1)
    {};
    using BaseDualSimplex::BaseDualSimplex;
    ~CudaDualSimplex();

    void initDualSimplex();
};


