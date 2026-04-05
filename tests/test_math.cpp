#include <gtest/gtest.h>

#include "mkl.h"

#include "../src/cpu/valuesVector.hpp"
#include "../src/cpu/matrix.hpp"
#include "../src/cpu/sequentialDualSimplex.hpp"

#include "../src/common/problem.hpp"


TEST(Math, Test1)
{
    int m = 3, n =3;
    ValuesVector rhs1(std::vector<double>{1.5, 2, 0.5});
    std::vector<double> expected = {0.5, 0.5, 1.75}
    ValuesVector sol1(n);
    Matrix A(
        std::vector<double>{1, 0.5, 2, 1, 1},
        std::vector<MKL_INT>{0, 2, 4, 5},
        std::vector<MKL_INT>{0, 1, 0, 2, 1},
        m, n
    );
    
    A.LUdecompose();
    A.solve(rhs1, sol1, true);
    
    EXPECT_THAT(sol1.getData(), ElementsAreArray(expected));
}


TEST(Math, Test2)
{
    int m = 3, n =3;
    ValuesVector rhs1(std::vector<double>{1.5, 2, 0.5});
    std::vector<double> expected = {1.25, 0.5, -0.5};
    ValuesVector sol1(n);
    Matrix A(
        std::vector<double>{1, 0.5, 2, 1, 1},
        std::vector<MKL_INT>{0, 2, 4, 5},
        std::vector<MKL_INT>{0, 1, 0, 2, 1},
        m, n
    );

    A.LUdecompose();
    A.solve(rhs1, sol1, false);
    
    EXPECT_THAT(sol1.getData(), ElementsAreArray(expected));
}


TEST(Math, Test3)
{
    int m = 3, n = 4;
    ValuesVector rhs1(std::vector<double>{1.5, 2, 0.5});
    ValuesVector rhs2(std::vector<double>{1.5, 2, 0.5, 1});
    ValuesVector sol1(m);
    ValuesVector sol2(m);
    ValuesVector sol3(m);
    ValuesVector sol4(n);
    Matrix A(
        std::vector<double>{1, 0.5, 4, 2, 1, 1, 1},
        std::vector<MKL_INT>{0, 3, 5, 7},
        std::vector<MKL_INT>{0, 1, 3, 0, 2, 1, 3},
        m, n
    );
    A.show();
    std::cout << "FULL_UPDATE" << std::endl;
    rhs2.show();
    rhs1.show();
    A.dotUpdate(
        rhs1, rhs1, 
        sol1, -1, 1, 
        IndexVector({0, 1, 3}), 
        SpmvOptions::UPDATE,
        false
    );
    sol1.show();

    std::cout << "SET_UPDATE" << std::endl;
    rhs2.show();
    A.dotUpdate(
        rhs2, rhs1, 
        sol2, -1, 1, 
        IndexVector({0, 1, 3}), 
        SpmvOptions::UPDATE,
        true
    );
    sol2.show();

    std::cout << "FULL_UPDATE_T" << std::endl;
    rhs1.show();
    rhs2.show();
    A.dotUpdate(
        rhs1, rhs2, 
        sol3, 1, 0, 
        IndexVector({0, 1, 3}), 
        SpmvOptions::UPDATE_T,
        false
    );
    sol3.show();

    std::cout << "SET_UPDATE_T" << std::endl;
    rhs2.show();
    A.dotUpdate(
        rhs1, rhs2, 
        sol4, -1, 1, 
        IndexVector({0, 1, 3}), 
        SpmvOptions::UPDATE_T,
        true
    );
    sol4.show();

}