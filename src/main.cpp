#include <CoinMpsIO.hpp>

#include "valuesVector.hpp"
#include "matrix.hpp"
#include "problem.hpp"
#include "sequentialDualSimplex.hpp"
#include "parser.hpp"

void test1();
void test2();
void test3();
void test4();


int main()
{
    Problem problem;
    std::string path = "../tests/test_data/afiro.mps";
    LPparser parser;
    parser.readMps(path, problem);
    // problem.show();
    sequentialDualSimplex solver(problem, "minInfeas");
    LPsolution solution = solver.solve("simple");
    solution.show();

    // test1();
    // test2();
    // test3(); 
    // test4();
}


void test1()
{
    size_t m = 3, n =5;
    ValuesVector costs(std::vector<double>{2, -1, 3, -2, 1});
    ValuesVector lower_range(std::vector<double>{1.5, 2, 0.5}); 
    ValuesVector upper_range(std::vector<double>{1.5, 2, 0.5}); 
    ValuesVector lower_bound(std::vector<double>{0, 0, 0, 0, 0}); 
    ValuesVector upper_bound(std::vector<double>{0, 0, 0, 0, 0}); 
    BoundaryTypeVector bound_type{
        BoundaryType::Lower, BoundaryType::Lower, BoundaryType::Lower,
        BoundaryType::Lower,BoundaryType::Lower}; 
    BoundaryTypeVector range_type{BoundaryType::Fixed, BoundaryType::Fixed, BoundaryType::Fixed}; 
    Matrix A(3, 5);
    A(0, 1) = 1;
    A(0, 2) = 0.5;
    A(0, 4) = 0.5;
    A(1, 2) = 1;
    A(1, 3) = 1;
    A(2, 0) = 1;
    A(2, 2) = -0.5;
    A(2, 4) = 0.5;
    Problem test_problem(
        bound_type, range_type, costs, lower_range, upper_range, 
        lower_bound, upper_bound, A); 
    sequentialDualSimplex solver(test_problem, "minInfeas");
    LPsolution solution = solver.solve("simple");

    LPsolution true_solution;
    true_solution.message = "analitic solution";
    true_solution.x = ValuesVector(std::vector<double>{0.5, 1.5, 0, 2, 0});
    true_solution.Z = -4.5;
    true_solution.solved = true;

    std::cout << "Test 1" << std::endl;
    std::cout << "numeric solution" << std::endl;
    solution.show();
    std::cout << "true solution" << std::endl;
    true_solution.show();
    std::cout << std::endl << std::endl; 
}


void test2()
{
    size_t m = 3, n =5;
    ValuesVector costs(std::vector<double>{-14, 5, -2, 1, -8});
    ValuesVector lower_range(std::vector<double>{5, 41, 15}); 
    ValuesVector upper_range(std::vector<double>{5, 41, 15}); 
    ValuesVector lower_bound(std::vector<double>{0, 0, 0, 0, 0}); 
    ValuesVector upper_bound(std::vector<double>{0, 0, 0, 0, 0}); 
    BoundaryTypeVector bound_type{
        BoundaryType::Lower, BoundaryType::Lower, BoundaryType::Lower,
        BoundaryType::Lower,BoundaryType::Lower}; 
    BoundaryTypeVector range_type{BoundaryType::Fixed, BoundaryType::Fixed, BoundaryType::Fixed}; 
    Matrix A(3, 5);
    A(0, 0) = 1;
    A(0, 1) = 1;
    A(0, 4) = -1;
    A(1, 0) = 5;
    A(1, 2) = 1;
    A(1, 4) = 3;
    A(2, 0) = -5;
    A(2, 3) = 1;
    A(2, 4) = 4;
    Problem test_problem(
        bound_type, range_type, costs, lower_range, upper_range, 
        lower_bound, upper_bound, A); 
    sequentialDualSimplex solver(test_problem, "minInfeas");
    LPsolution solution = solver.solve("simple");

    LPsolution true_solution;
    true_solution.message = "analitic solution";
    true_solution.x = ValuesVector(std::vector<double>{7, 0, 0, 42, 2});
    true_solution.Z = -72;
    true_solution.solved = true;

    std::cout << "Test 2" << std::endl;
    std::cout << "numeric solution" << std::endl;
    solution.show();
    std::cout << "true solution" << std::endl;
    true_solution.show();
    std::cout << std::endl << std::endl; 
}


void test3()
{
    size_t m = 2, n = 3;
    ValuesVector costs(std::vector<double>{3, 2, 3});
    ValuesVector lower_range(std::vector<double>{0, 8}); 
    ValuesVector upper_range(std::vector<double>{2, 0}); 
    ValuesVector lower_bound(std::vector<double>{0, 0, 1}); 
    ValuesVector upper_bound(std::vector<double>{0, 0, 0}); 
    BoundaryTypeVector bound_type{BoundaryType::Lower, BoundaryType::Lower, BoundaryType::Lower}; 
    BoundaryTypeVector range_type{BoundaryType::Upper, BoundaryType::Lower}; 
    Matrix A(2, 3);
    A(0, 0) = 2;
    A(0, 1) = 1;
    A(0, 2) = 1;
    A(1, 0) = 3;
    A(1, 1) = 8;
    A(1, 2) = 2;
    Problem test_problem(
        bound_type, range_type, costs, lower_range, upper_range, 
        lower_bound, upper_bound, A); 
    sequentialDualSimplex solver(test_problem, "minInfeas");
    LPsolution solution = solver.solve("simple");

    LPsolution true_solution;
    true_solution.message = "analitic solution";
    true_solution.x = ValuesVector(std::vector<double>{0, 0.75, 1, 0.25, 0});
    true_solution.Z = 4.5;
    true_solution.solved = true;

    std::cout << "Test 3" << std::endl;
    std::cout << "numeric solution" << std::endl;
    solution.show();
    std::cout << "true solution" << std::endl;
    true_solution.show();
    std::cout << std::endl << std::endl; 
}


void test4()
{
    size_t m = 3, n = 5;
    ValuesVector costs(std::vector<double>{-2, -3, 0, 0, 0});
    ValuesVector lower_range(std::vector<double>{5, 9, 4}); 
    ValuesVector upper_range(std::vector<double>{5, 9, 4}); 
    ValuesVector lower_bound(std::vector<double>{0, 0, 0, 0, 0}); 
    ValuesVector upper_bound(std::vector<double>{0, 0, 0, 0, 0}); 
    BoundaryTypeVector bound_type{BoundaryType::Lower, BoundaryType::Lower, BoundaryType::Lower, 
                                                       BoundaryType::Lower, BoundaryType::Lower}; 
    BoundaryTypeVector range_type{BoundaryType::Fixed, BoundaryType::Fixed, BoundaryType::Fixed}; 
    Matrix A(3, 5);
    A(0, 0) = 1;
    A(0, 1) = 1;
    A(0, 2) = 1;
    A(1, 0) = 1;
    A(1, 1) = 3;
    A(1, 3) = 1;
    A(2, 0) = 1;
    A(2, 4) = 1;
    Problem test_problem(
        bound_type, range_type, costs, lower_range, upper_range, 
        lower_bound, upper_bound, A); 
    sequentialDualSimplex solver(test_problem, "minInfeas");
    LPsolution solution = solver.solve("simple");

    LPsolution true_solution;
    true_solution.message = "analitic solution";
    true_solution.x = ValuesVector(std::vector<double>{3, 2, 0, 0, 1});
    true_solution.Z = -12;
    true_solution.solved = true;

    std::cout << "Test 4" << std::endl;
    std::cout << "numeric solution" << std::endl;
    solution.show();
    std::cout << "true solution" << std::endl;
    true_solution.show();
    std::cout << std::endl << std::endl; 
}




