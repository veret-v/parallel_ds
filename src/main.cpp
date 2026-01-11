#include <CoinMpsIO.hpp>

#include "valuesVector.hpp"
#include "matrix.hpp"
#include "problem.hpp"
#include "sequentialDualSimplex.hpp"
#include "parallelDualSimplex.hpp"
#include "parser.hpp"


int main()
{
    std::string path = "../tests/test_data/afiro.mps";
    LPparser parser;

    Problem problem;
    parser.readMps(path, problem);
    ParallelDualSimplex solver(problem);
    solver.presolve("minInfeas");
    LPsolution solution = solver.solve("elaborated");
    solution.show();

    Problem problem1;
    parser.readMps(path, problem1);
    SequentialDualSimplex solver1(problem1);
    solver1.presolve("minInfeas");
    LPsolution solution1 = solver1.solve("elaborated");
    solution1.show();
}

