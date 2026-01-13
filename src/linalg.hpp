#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_set>

#include "matrix.hpp"
#include "valuesVector.hpp"
#include "types.hpp"

#define EPS_ZERO 1e-8
#define EPS_SWAP 1e-6


namespace linalg
{
    Matrix ones(const size_t n);
    ValuesVector unit(const size_t size, const size_t p);

    #ifdef WITH_CUDA
        ValuesVector PFIsolve(const std::vector<EtaMatrix>& A, const ValuesVector& b, const bool transpose);
        bool PFIdecompose(const Matrix& A, std::vector<EtaMatrix>& decomposed);
        
        // class DevEtaMatrixVector
        // {
        // friend ValuesVector PFIsolve(const DevEtaMatrixVector& A, const ValuesVector& b, const bool transpose);

        // protected:
        //     double* device_mem_vecs = nullptr;
        //     size_t* device_mem_ids  = nullptr;

        //     size_t capacity = 0;
        //     size_t size     = 0;
        //     size_t eta_size = 0;

        //     void allocateMemory();
        //     void freeMemory();

        // public:
        //     DevEtaMatrixVector(const size_t _eta_size);
        //     ~DevEtaMatrixVector();

        //     void clear();
        //     void pushBack(const ValuesVector& values_vector, const size_t idx);
            
        // };
    #else
        ValuesVector PFIsolve(const std::vector<EtaMatrix>& A, const ValuesVector& b, const bool transpose);
        bool PFIdecompose(const Matrix& A, std::vector<EtaMatrix>& decomposed);
        bool checkPFIdecompose(const std::vector<EtaMatrix>& B_eta_repr, const Matrix& B);
        void dotEtaMatrix(const EtaMatrix& eta_matrix, Matrix& matrix);
        ValuesVector LUsolve(const std::vector<EtaMatrix>& A, const ValuesVector& b, const bool transpose);
        bool LUdecompose(const Matrix& A, std::vector<EtaMatrix>& decomposed);
    #endif
    
} // namespace linalg

