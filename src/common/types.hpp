#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <typeinfo>

#include "../cpu/valuesVector.hpp"


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
    UNKNOWN
};

typedef std::vector<int> IndexVector;
typedef std::vector<BoundaryType> BoundaryTypeVector;
typedef int cudaVectorSize;
typedef std::tuple <int, int> MatrixSize;
typedef int VectorSize;
typedef std::tuple<ValuesVector, int> EtaMatrix;

