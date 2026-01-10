#include "linalg.hpp"



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
        for(start; start != end; start += 1)
        {
            size_t k = std::get<1>(*start);
            ValuesVector eta_val =  std::get<0>(*start);
            ValuesVector buff_result(result);
            for (size_t i = 0; i < b.getSize(); i++)
                buff_result[i] = (i != k) ? result[i] + result[k] * eta_val[i] : result[k] * eta_val[k];    
            result = buff_result;        
        }
    } else
    {
        auto start = A.rbegin();
        auto end = A.rend();
        for(start; start != end; start += 1)
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
    bool is_identity;

    for (size_t i = 0; i < m; i++)
    {
        is_identity = true;
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
            ValuesVector bs_eta_matrix_val(m);
            bs_eta_matrix_val[swap_id] = -1; 
            ValuesVector sgn_eta_matrix_val(m);
            bs_eta_matrix_val[swap_id] = -1; 
            
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
        Matrix new_buff_A(buff_A);
        for (size_t j = 0; j < m; j++)
        { 
            eta_matrix_val[j] = (j != i) ? - buff_A(j, i) / buff_A(i, i) : 1 / buff_A(i, i); 
            for (size_t k = 0; k < m; k++)
                new_buff_A(j, k) = (j != i) ? buff_A(j, k) + buff_A(i, k) * eta_matrix_val[j] : buff_A(i, k) * eta_matrix_val[j];
        }   
        buff_A = new_buff_A;
        decomposed.push_back(EtaMatrix(eta_matrix_val, i));
    }
    return true;
}


bool linalg::checkPFIdecompose(const std::vector<EtaMatrix>& B_eta_repr, const Matrix& B)
{
    Matrix buff_B(B);
    auto start = B_eta_repr.begin();
    auto end = B_eta_repr.end();

    size_t m = std::get<1>(B.getSize());
    for(start; start != end; start += 1)
        dotEtaMatrix(*start, buff_B);
       
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j= 0; j < m; j++)
            if (!(buff_B(i, j) == 0 && i != j || buff_B(i, j) == 1 && i == j )) return  false;
    }
    return true;
}


void linalg::dotEtaMatrix(const EtaMatrix& eta_matrix, Matrix& matrix)
{
    size_t m = std::get<1>(matrix.getSize());
    size_t k = std::get<1>(eta_matrix);
    ValuesVector eta_val =  std::get<0>(eta_matrix);
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j= 0; j < m; j++)
            matrix(i, j) = (i != k) ? matrix(i, j) + matrix(k, j) * eta_val[i] : matrix(k, j) * eta_val[i];
    }
    matrix.show();
}


ValuesVector linalg::LUsolve(const std::vector<EtaMatrix>& A, const ValuesVector& b, const bool transpose)
{
    return ValuesVector();
}


bool linalg::LUdecompose(const Matrix& A, std::vector<EtaMatrix>& decomposed)
{
    size_t m = std::get<0>(A.getSize());
    Matrix U = A;
    Matrix per_rows(m, m);
    Matrix per_cols(m, m);
    std::vector<EtaMatrix> L;
    std::unordered_set<size_t> P(m);
    std::unordered_set<size_t>  Q(m);

    for (size_t i = 0; i < m; i++)
    {
        P.insert(i);
        Q.insert(i);
    }

    for (size_t k = 0; k < m; k++)
    {
        size_t p, q;
        for (size_t p : P)
            for (size_t q : Q)
                if (U(p, q) != 0) break;
        
        per_rows(p, k) = 1;
        per_cols(k, q) = 1;

        P.erase(P.find(p));
        Q.erase(Q.find(q));


        ValuesVector L_tmp(m);
        L_tmp[p] = 1;
        for (auto i : P)
        {
            if (U(i, q) != 0)
            {
               L_tmp[i] = -U(i, q) / U(p, q);
               U(i, q) = 0;
            }
            
            for (auto j : Q)
                U(i, j) += L_tmp[i] * U(p, j);
        }
        L.push_back(EtaMatrix(L_tmp, p));
    }
    return true;
    
    
}
