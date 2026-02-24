#include "linalg.hpp"


ValuesVector linalg::unit(const int size, const int p)
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
            int k = std::get<1>(*start);
            ValuesVector eta_val =  std::get<0>(*start);
            ValuesVector buff_result(result);
            for (int i = 0; i < b.getSize(); i++)
                buff_result[i] = (i != k) ? result[i] + result[k] * eta_val[i] : result[k] * eta_val[k];    
            result = buff_result;        
        }
    } else
    {
        auto start = A.rbegin();
        auto end = A.rend();
        for(start; start != end; start += 1)
        {
            int k = std::get<1>(*start);
            ValuesVector eta_val = std::get<0>(*start);
            result[k] = eta_val.dot(result);  
        }
    }

    return result;
}


bool linalg::PFIdecompose(const Matrix& A, std::vector<EtaMatrix>& decomposed)
{
    Matrix buff_A(A);
    buff_A.genCSRorder();
    int m = std::get<0>(buff_A.getSize());
    bool is_identity;

    for (int i = 0; i < m; i++)
    {
        is_identity = true;
        if (fabs(buff_A(i, i)) < EPS_ZERO)
        {
            int swap_id  = i;
            for (int k = i; k < m; k++)
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
            sgn_eta_matrix_val[swap_id] = -1; 
            
            buff_A.swapRows(i, swap_id);
            
            decomposed.push_back(EtaMatrix(fs_eta_matrix_val, swap_id));
            decomposed.push_back(EtaMatrix(bs_eta_matrix_val, i));    
            decomposed.push_back(EtaMatrix(sgn_eta_matrix_val, swap_id));    
        }
       
        ValuesVector eta_matrix_val(m);
        for (int j = 0; j < m; j++)
            eta_matrix_val[j] = (j != i) ? - buff_A(j, i) / buff_A(i, i) : 1 / buff_A(i, i); 
        EtaMatrix eta = EtaMatrix(eta_matrix_val, i);
        
        decomposed.push_back(eta);
        buff_A.dotEtaMatrix(eta);
    }
    return true;
}


// ValuesVector linalg::LUsolve(const std::vector<EtaMatrix>& A, const ValuesVector& b, const bool transpose)
// {
//     return ValuesVector();
// }


// bool linalg::LUdecompose(const Matrix& A, std::vector<EtaMatrix>& decomposed)
// {
//     int m = std::get<0>(A.getSize());
//     Matrix U = A;
//     Matrix per_rows(m, m);
//     Matrix per_cols(m, m);
//     std::vector<EtaMatrix> L;
//     std::unordered_set<int> P;
//     std::unordered_set<int> Q;

//     for (int i = 0; i < m; i++)
//     {
//         P.insert(i);
//         Q.insert(i);
//     }

//     for (int k = 0; k < m; k++)
//     {
//         int p, q;
//         for (int p : P)
//             for (int q : Q)
//                 if (U(p, q) != 0) break;
        
//         per_rows(p, k) = 1;
//         per_cols(k, q) = 1;

//         P.erase(P.find(p));
//         Q.erase(Q.find(q));


//         ValuesVector L_tmp(m);
//         L_tmp[p] = 1;
//         for (auto i : P)
//         {
//             if (U(i, q) != 0)
//             {
//                L_tmp[i] = -U(i, q) / U(p, q);
//                U(i, q) = 0;
//             }
            
//             for (auto j : Q)
//                 U(i, j) += L_tmp[i] * U(p, j);
//         }
//         L.push_back(EtaMatrix(L_tmp, p));
//     }
//     return true;
    
    
// }
