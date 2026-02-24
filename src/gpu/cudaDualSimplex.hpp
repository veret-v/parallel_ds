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

#define EPS_BOUND 1e-10
#define EPS_ALPHA 1e-8
#define EPS_D     1e-7
#define EPS_COSTS 1e-8
#define EPS_A     1e-5
#define EPS_Z     1e-12
#define REFACT_ERR 1e-9

#define PERTURB_RATIO 0.25
#define PSI           1e-5
#define MAX_CYCLE     5
#define REFACT_FREQ   200


class CudaDualSimplex : public BaseDualSimplex<CudaSparseMatrix, CudaDataDenseVector, CudaIndexVector>
{
protected:
    PFIfactor pfi_factor;

    CudaDataDenseVector beta;
    
    cublasHandle_t cu_handle;
    cusparseHandle_t sp_handle;

    cudssHandle_t cudss_handle;
    cudssConfig_t cudss_config;
    cudssData_t cudss_data;

    cudssHandle_t cudss_handle_T;
    cudssConfig_t cudss_config_T;
    cudssData_t cudss_data_T;

    void initDualSimplex() override;

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
    using BaseDualSimplex::BaseDualSimplex;

};


