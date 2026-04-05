#include "linalg.hpp"


ValuesVector linalg::unit(const int size, const int p)
{
    ValuesVector new_vec(size);
    new_vec[p] = 1;
    return new_vec;
}


void linalg::PFIsolve(
    const std::vector<EtaMatrix>& A, 
    const ValuesVector& b, 
    ValuesVector& sol, 
    const bool transpose
)
{
    sol = b;
    if (!transpose)
    {
        auto start = A.begin();
        auto end = A.end();
        for(start; start != end; start += 1)
        {
            int k = std::get<1>(*start);
            ValuesVector eta_val =  std::get<0>(*start);
            ValuesVector buff_result(sol);
            for (int i = 0; i < b.getSize(); i++)
                buff_result[i] = (i != k) ? sol[i] + sol[k] * eta_val[i] : sol[k] * eta_val[k];    
            sol = buff_result;        
        }
    } 
    else
    {
        auto start = A.rbegin();
        auto end = A.rend();
        for(start; start != end; start += 1)
        {
            int k = std::get<1>(*start);
            ValuesVector eta_val = std::get<0>(*start);
            sol[k] = eta_val.dot(sol);  
        }
    }

}

