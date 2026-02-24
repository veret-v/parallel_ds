#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <typeinfo>

#include "../cpu/valuesVector.hpp"
#include "../cpu/matrix.hpp"
#include "../gpu/cudaSparseMatrix.hpp"
#include "../gpu/cudaDenseVector.hpp"


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
    simpleRatio,
    elaboratedMethod,
    UNKNOWN
};

enum class PresolverMethods
{
    minDualInfeasibility,
    panMethod,
    UNKNOWN
};

enum class SpmvOptions
{
    FULL_UPDATE,
    SET_UPDATE,
    FULL_UPDATE_T,
    SET_UPDATE_T,
    UNKNOWN
};

typedef std::vector<int> IndexVector;
typedef std::tuple <int, int> cudaMatrixSize;
typedef std::vector<BoundaryType> BoundaryTypeVector;
typedef int cudaVectorSize;
typedef std::tuple <int, int> MatrixSize;
typedef int VectorSize;
typedef std::tuple<ValuesVector, int> EtaMatrix;

