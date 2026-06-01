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
    int m = b.getSize();
    if (!transpose)
    {
        auto start = A.begin();
        auto end = A.end();
        for(start; start != end; start += 1)
        {
            int k = std::get<1>(*start);
            ValuesVector eta_val =  std::get<0>(*start);
            double sk = sol[k];
            cblas_daxpy(m, sk, eta_val.getPointerData(), 1, sol.getPointerData(), 1); 
            sol[k] = sk * eta_val[k];      
        }
    } 
    else
    {
        auto start = A.rbegin();
        auto end = A.rend();
        for(start; start != end; start += 1)
        {
            int k = std::get<1>(*start);
            ValuesVector eta_val =  std::get<0>(*start);
            sol[k] = cblas_ddot(m, eta_val.getPointerData(), 1, sol.getPointerData(), 1);
        }   
    }

}


void linalg::APFsolve(
    ValuesVector& apf_values, 
    ValuesVector& rho_values, 
    ValuesVector& sol, 
    bool transpose
)
{
    int m = sol.getSize();
    if (transpose)
    {
        double y = -cblas_ddot(m, apf_values.getPointerData(), 1, sol.getPointerData(), 1);
        cblas_daxpy(m, y, rho_values.getPointerData(), 1, sol.getPointerData(), 1); 
    } 
    else
    {
        throw "no method provided";
    }

}