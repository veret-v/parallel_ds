#pragma once

#include <typeinfo>
#include <tuple>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <cusolverDn.h>
#include <cusolverRf.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <math.h>
#include <cudss.h>

#include "utillities.hpp"
#include "cudaDataDenseVector.hpp"

#include "../common/types.hpp"

#define SING_EPS     1e-12

class PFIfactor
{
private:
    int size     = 0;
    int capacity = 0;
    int col_len  = 0;

    double* device_values = nullptr;
    int* device_col_id    = nullptr;

    void allocateMemory();
    void freeMemory();

public:
    PFIfactor(const int col_len, const int capacity);
    ~PFIfactor();

    void addEtaMatrix(const int q, CudaDataDenseVector& vals);
    void applyPFI(
        const cublasHandle_t& handle, 
        const CudaDataDenseVector& rhs, 
        CudaDataDenseVector& sol, 
        const bool& transpose
    );
    void resetPFI() {size = 0;};

    int getSize() {return size;};
};