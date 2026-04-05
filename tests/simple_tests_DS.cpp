#include <gtest/gtest.h>

#include "mkl.h"

#include "../src/cpu/valuesVector.hpp"
#include "../src/cpu/matrix.hpp"
#include "../src/cpu/sequentialDualSimplex.hpp"

#include "../src/common/problem.hpp"


TEST(DualSimplexSimple, Test1)
{
    int m = 3, n =5;
    ValuesVector costs(std::vector<double>{2, -1, 3, -2, 1});
    ValuesVector lower_range(std::vector<double>{1.5, 2, 0.5}); 
    ValuesVector upper_range(std::vector<double>{1.5, 2, 0.5}); 
    ValuesVector lower_bound(std::vector<double>{0, 0, 0, 0, 0}); 
    ValuesVector upper_bound(std::vector<double>{0, 0, 0, 0, 0}); 
    BoundaryTypeVector bound_type{
        BoundaryType::Lower, BoundaryType::Lower, BoundaryType::Lower,
        BoundaryType::Lower,BoundaryType::Lower}; 
    BoundaryTypeVector range_type{BoundaryType::Fixed, BoundaryType::Fixed, BoundaryType::Fixed}; 
    Matrix A(
        std::vector<double>{1, 0.5, 0.5, 1, 1, 1, -0.5, 0.5},
        std::vector<MKL_INT>{0, 3, 5, 8},
        std::vector<MKL_INT>{1, 2, 4, 2, 3, 0, 2, 4},
        m, n
    );
    Problem test_problem(
        bound_type, range_type, costs, lower_range, upper_range, 
        lower_bound, upper_bound, A); 
    SequentialDualSimplex solver(test_problem);
    solver.initDualSimplex();
    solver.presolve("minInfeas");
    LPsolution solution = solver.solve("simple");

    LPsolution true_solution;
    true_solution.message = "analitic solution";
    true_solution.x = ValuesVector(std::vector<double>{0.5, 1.5, 0, 2, 0});
    true_solution.Z = -4.5;
    true_solution.solved = true;
    EXPECT_NEAR(solution.Z, true_solution.Z, 1e-8);
}


TEST(DualSimplexSimple, Test2)
{
    int m = 3, n =5;
    ValuesVector costs(std::vector<double>{-14, 5, -2, 1, -8});
    ValuesVector lower_range(std::vector<double>{5, 41, 15}); 
    ValuesVector upper_range(std::vector<double>{5, 41, 15}); 
    ValuesVector lower_bound(std::vector<double>{0, 0, 0, 0, 0}); 
    ValuesVector upper_bound(std::vector<double>{0, 0, 0, 0, 0}); 
    BoundaryTypeVector bound_type{
        BoundaryType::Lower, BoundaryType::Lower, BoundaryType::Lower,
        BoundaryType::Lower,BoundaryType::Lower}; 
    BoundaryTypeVector range_type{BoundaryType::Fixed, BoundaryType::Fixed, BoundaryType::Fixed}; 
    Matrix A(
        std::vector<double>{1, 1, -1, 5, 1, 3, -5, 1, 4},
        std::vector<MKL_INT>{0, 3, 6, 9},
        std::vector<MKL_INT>{0, 1, 4, 0, 2, 4, 0, 3, 4},
        m, n
    );
    Problem test_problem(
        bound_type, range_type, costs, lower_range, upper_range, 
        lower_bound, upper_bound, A); 
    SequentialDualSimplex solver(test_problem);
    solver.initDualSimplex();
    solver.presolve("minInfeas");
    LPsolution solution = solver.solve("simple");

    LPsolution true_solution;
    true_solution.message = "analitic solution";
    true_solution.x = ValuesVector(std::vector<double>{7, 0, 0, 42, 2});
    true_solution.Z = -72;
    true_solution.solved = true;
    EXPECT_NEAR(solution.Z, true_solution.Z, 1e-8);
}


TEST(DualSimplexSimple, Test3)
{
    int m = 2, n = 3;
    ValuesVector costs(std::vector<double>{3, 2, 3});
    ValuesVector lower_range(std::vector<double>{0, 8}); 
    ValuesVector upper_range(std::vector<double>{2, 0}); 
    ValuesVector lower_bound(std::vector<double>{0, 0, 1}); 
    ValuesVector upper_bound(std::vector<double>{0, 0, 0}); 
    BoundaryTypeVector bound_type{BoundaryType::Lower, BoundaryType::Lower, BoundaryType::Lower}; 
    BoundaryTypeVector range_type{BoundaryType::Upper, BoundaryType::Lower}; 
    Matrix A(
        std::vector<double>{2, 1, 1, 3, 8, 2},
        std::vector<MKL_INT>{0, 3, 6},
        std::vector<MKL_INT>{0, 1, 2, 0, 1, 2},
        m, n
    );
    Problem test_problem(
        bound_type, range_type, costs, lower_range, upper_range, 
        lower_bound, upper_bound, A); 
    SequentialDualSimplex solver(test_problem);
    solver.initDualSimplex();
    solver.presolve("minInfeas");
    LPsolution solution = solver.solve("simple");

    LPsolution true_solution;
    true_solution.message = "analitic solution";
    true_solution.x = ValuesVector(std::vector<double>{0, 0.75, 1, 0.25, 0});
    true_solution.Z = 4.5;
    true_solution.solved = true;
    EXPECT_NEAR(solution.Z, true_solution.Z, 1e-8);
}


TEST(DualSimplexSimple, Test4)
{
    int m = 3, n = 5;
    ValuesVector costs(std::vector<double>{-2, -3, 0, 0, 0});
    ValuesVector lower_range(std::vector<double>{5, 9, 4}); 
    ValuesVector upper_range(std::vector<double>{5, 9, 4}); 
    ValuesVector lower_bound(std::vector<double>{0, 0, 0, 0, 0}); 
    ValuesVector upper_bound(std::vector<double>{0, 0, 0, 0, 0}); 
    BoundaryTypeVector bound_type{BoundaryType::Lower, BoundaryType::Lower, BoundaryType::Lower, 
                                                       BoundaryType::Lower, BoundaryType::Lower}; 
    BoundaryTypeVector range_type{BoundaryType::Fixed, BoundaryType::Fixed, BoundaryType::Fixed}; 
    Matrix A(
        std::vector<double>{1, 1, 1, 1, 3, 1, 1, 1},
        std::vector<MKL_INT>{0, 3, 6, 8},
        std::vector<MKL_INT>{0, 1, 2, 0, 1, 3, 0, 4},
        m, n
    );
    Problem test_problem(
        bound_type, range_type, costs, lower_range, upper_range, 
        lower_bound, upper_bound, A); 
    SequentialDualSimplex solver(test_problem);
    solver.initDualSimplex();
    solver.presolve("minInfeas");
    LPsolution solution = solver.solve("simple");

    LPsolution true_solution;
    true_solution.message = "analitic solution";
    true_solution.x = ValuesVector(std::vector<double>{3, 2, 0, 0, 1});
    true_solution.Z = -12;
    true_solution.solved = true;
    EXPECT_NEAR(solution.Z, true_solution.Z, 1e-4);
}