#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <typeinfo>

#include "./cpu_version/valuesVector.hpp"
#include "./cpu_version/matrix.hpp"
#include "./cuda_version/cudaSparseMatrix.hpp"
#include "./cuda_version/cudaDenseVector.hpp"


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

typedef std::vector<size_t> IndexVector;
typedef std::tuple <size_t, size_t> cudaMatrixSize;
typedef std::vector<BoundaryType> BoundaryTypeVector;
typedef size_t cudaVectorSize;
typedef std::tuple <size_t, size_t> MatrixSize;
typedef size_t VectorSize;
typedef std::tuple<ValuesVector, size_t> EtaMatrix;

