#include "linalg.hpp"


// void linalg::DevEtaMatrixVector::allocateMemory()
// {
//     cudaMalloc(&device_mem_vecs, capacity*eta_size*sizeof(double));
//     cudaMalloc(&device_mem_ids, capacity*sizeof(size_t));
// }


// void linalg::DevEtaMatrixVector::freeMemory()
// {
//     cudaFree(device_mem_vecs);
//     cudaFree(device_mem_ids);
// }


// linalg::DevEtaMatrixVector::~DevEtaMatrixVector()
// {
//     freeMemory();
// }


// linalg::DevEtaMatrixVector::DevEtaMatrixVector(const size_t _eta_size)
// {
//     eta_size = _eta_size;
//     size     = _eta_size;
//     capacity = 2 * size;

//     allocateMemory();
// }


// void linalg::DevEtaMatrixVector::clear()
// {
//     size = 0;
//     capacity = 0;
//     cudaFree(device_mem_vecs);
//     cudaFree(device_mem_ids);
// }


// void linalg::DevEtaMatrixVector::pushBack(const ValuesVector& values_vector, const size_t idx)
// {
//     if (size + 1 < capacity)
//     {
//         cudaMemcpy(device_mem_vecs + size * eta_size, values_vector.host_mem, eta_size*sizeof(double), cudaMemcpyDefault);
//         cudaMemcpy(device_mem_ids + size, &idx, sizeof(size_t), cudaMemcpyDefault);
//         size += 1;
//     }
//     else
//     {
//         double* buff_vec;
//         size_t* buff_idx;

//         cudaMallocHost(&buff_vec, size * eta_size * sizeof(double));
//         cudaMallocHost(&buff_idx, size * sizeof(size_t));
//         cudaMemcpy(buff_vec, device_mem_vecs, size * eta_size * sizeof(double), cudaMemcpyDefault);
//         cudaMemcpy(buff_idx, device_mem_ids, size * sizeof(size_t), cudaMemcpyDefault);
//         freeMemory();

//         capacity = 2 * size;

//         allocateMemory();
//         cudaMemcpy(device_mem_vecs, buff_vec, size * eta_size * sizeof(double), cudaMemcpyDefault);
//         cudaMemcpy(device_mem_ids, buff_idx, size * sizeof(size_t), cudaMemcpyDefault);
//         cudaMemcpy(device_mem_vecs + size * eta_size, values_vector.host_mem, eta_size*sizeof(double), cudaMemcpyDefault);
//         cudaMemcpy(device_mem_ids + size, &idx, sizeof(size_t), cudaMemcpyDefault);

//         size += 1;
//     }
// }


Matrix linalg::ones(const size_t n)
{
    Matrix ones(n, n);
    for (size_t i = 0; i < n; i++)
        ones(i, i) = 1;
    return ones;
}


ValuesVector linalg::unit(const size_t size, const size_t p)
{
    ValuesVector new_vec(size);
    new_vec[p] = 1;
    return new_vec;
}


ValuesVector linalg::PFIsolve(const std::vector<EtaMatrix>& A, const ValuesVector& b, const bool transpose)
{
    ValuesVector result = b;
    if (!transpose)
    {
        auto start = A.begin();
        auto end = A.end();
        for(auto start = A.begin(); start != A.end(); start += 1)
        {
            size_t k = std::get<1>(*start);
            ValuesVector eta_val =  std::get<0>(*start);
            result.PFIupdate(eta_val, k);   
        }
    } else
    {
        for(auto start = A.rbegin(); start != A.rend(); start += 1)
        {
            size_t k = std::get<1>(*start);
            ValuesVector eta_val = std::get<0>(*start);
            result[k] = eta_val.dot(result);  
        }
    }

    return result;
}


bool linalg::PFIdecompose(const Matrix& A, std::vector<EtaMatrix>& decomposed)
{
    Matrix buff_A(A);
    size_t m = std::get<0>(buff_A.getSize());

    for (size_t i = 0; i < m; i++)
    {
        if (fabs(buff_A(i, i)) < EPS_ZERO)
        {
            size_t swap_id  = i;
            for (size_t k = i; k < m; k++)
            {

                if (fabs(buff_A(k, i)) > EPS_SWAP) 
                {
                    swap_id = k;
                    break;
                }
            }

            if (swap_id == i) 
            {
                decomposed.clear();
                return false;
            }

            ValuesVector fs_eta_matrix_val(m);
            fs_eta_matrix_val[i] = 1; 
            fs_eta_matrix_val[swap_id] = 1; 
            ValuesVector bs_eta_matrix_val(m);
            bs_eta_matrix_val[swap_id] = -1; 
            bs_eta_matrix_val[i] = 1; 
            ValuesVector sgn_eta_matrix_val(m);
            sgn_eta_matrix_val[swap_id] = -1; 
            
            for (size_t k = 0; k < m; k++)
            {
                buff_A(i, k) = buff_A(i, k) + buff_A(swap_id, k);
                buff_A(swap_id, k) = -buff_A(swap_id, k) + buff_A(i, k);
            }
            
            decomposed.push_back(EtaMatrix(fs_eta_matrix_val, swap_id));
            decomposed.push_back(EtaMatrix(bs_eta_matrix_val, i));    
            decomposed.push_back(EtaMatrix(sgn_eta_matrix_val, swap_id));   
        }
       
        ValuesVector eta_matrix_val(m);
        for (size_t j = 0; j < m; j++)
        { 
            eta_matrix_val[j] = (j != i) ? - buff_A(j, i) / buff_A(i, i) : 1 / buff_A(i, i); 
            for (size_t k = 0; k < m; k++)
                buff_A(j, k) = (j != i) ? buff_A(j, k) + buff_A(i, k) * eta_matrix_val[j] : buff_A(i, k);
        }   
        for (size_t k = 0; k < m; k++)
            buff_A(i, k) = buff_A(i, k) * eta_matrix_val[i];
            
        decomposed.push_back(EtaMatrix(eta_matrix_val, i));
    }
    return true;
}
