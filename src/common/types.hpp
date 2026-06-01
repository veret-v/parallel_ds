#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <typeinfo>

#include "../cpu/valuesVector.hpp"


enum class ALgorithmPart
{
    Init,
    Factor,
    RestoreProc,
    Btran,
    PivotRow,
    RatioTestPart1,
    RatioTestPart2,
    Ftran,
    FtranBfrt,
    FtranBeta,
    Pricing,
    UpdateRedCosts,
    BasisUpate
};


enum MpiTag 
{ 
    BTran, 
    FTran_BFRT, 
    TransferBasisChange,
    SolutionStatus,
    Refact 
};


enum class PresolveActionType 
{ 
    FIX_VAR, 
    REMOVE_COL, 
    SCALE 
};


enum class BoundaryType
{
    Fixed,
    Free,
    Boxed,
    Upper,
    Lower,
    UNKNOWN
};

enum class SolverMethods
{
    elaboratedMethod,
    UNKNOWN
};

enum class Phase1OutStatus
{
    Solved,
    DualInfeas,
    NeedRestart
};

enum class PresolverMethods
{
    minDualInfeasibility,
    UNKNOWN
};

enum class SpmvOptions
{
    UPDATE,
    UPDATE_T,
    UPDATE_T_SP_ROW_WISE,
    UPDATE_T_SP_COL_WISE,
    UNKNOWN
};

typedef std::vector<int> IndexVector;
typedef std::vector<BoundaryType> BoundaryTypeVector;
typedef int cudaVectorSize;
typedef std::tuple <int, int> MatrixSize;
typedef int VectorSize;
typedef std::tuple<ValuesVector, int> EtaMatrix;



struct Candidate
{
    int p_idx_  = 0;
    int p_      = 0;
    int q_idx_  = 0;
    int q_      = 0;

    bool is_lower_  = false;
    bool bfrt_done_ = false;
    bool is_active_ = false;

    double delta_  = 0.0;
    double weight_ = 0.0;
    double x_p_     = 0.0;

    ValuesVector tau_;
    ValuesVector rho_;
    ValuesVector alpha_q_;
    ValuesVector column_change_;
    ValuesVector delta_xB_;
    ValuesVector APF_update_;

    Candidate(
        int p_idx, int p, int q_idx, int q,
        bool is_lower, bool bfrt_done, bool is_active,
        double delta, double weight,
        ValuesVector tau, ValuesVector rho, ValuesVector alpha_q,
        ValuesVector column_change, ValuesVector delta_xB, ValuesVector APF_update
    ) : 
    p_idx_(p_idx), p_(p), q_idx_(q_idx), q_(q),
    is_lower_(is_lower), bfrt_done_(bfrt_done), is_active_(is_active),
    delta_(delta), weight_(weight),
    tau_(std::move(tau)), rho_(std::move(rho)), alpha_q_(std::move(alpha_q)),
    column_change_(std::move(column_change)), delta_xB_(std::move(delta_xB)),
    APF_update_(std::move(APF_update))
    {}
};

