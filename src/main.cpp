#include <CoinMpsIO.hpp>

#include "valuesVector.hpp"
#include "matrix.hpp"
#include "problem.hpp"
#include "sequentialDualSimplex.hpp"
#include "parser.hpp"


int main()
{
    std::string path = "../tests/test_data/adlittle.mps";
    LPparser parser;

    Problem problem;
    parser.readMps(path, problem);
    sequentialDualSimplex solver(problem, "minInfeas");
    LPsolution solution = solver.solve("elaborated");
    solution.show();

    Problem problem1;
    parser.readMps(path, problem1);
    sequentialDualSimplex solver1(problem1, "minInfeas");
    LPsolution solution1 = solver1.solve("simple");
    solution1.show();
}

