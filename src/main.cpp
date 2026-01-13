#include <CoinMpsIO.hpp>

#include <filesystem>

#include "valuesVector.hpp"
#include "matrix.hpp"
#include "problem.hpp"
#include "sequentialDualSimplex.hpp"
#include "parallelDualSimplex.hpp"
#include "parser.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    std::string file_name = argv[1];
    fs::path base_path = "../tests/test_data/";
    fs::path path = base_path / file_name;

    LPparser parser;

    Problem problem;
    parser.readMps(path, problem);
    ParallelDualSimplex solver(problem);
    solver.presolve("minInfeas");
    LPsolution solution = solver.solve("elaborated");
    
    Problem problem1;
    parser.readMps(path, problem1);
    SequentialDualSimplex solver1(problem1);
    solver1.presolve("minInfeas");
    LPsolution solution1 = solver1.solve("elaborated");

    std::cout << "Parallel:" << std::endl;
    solution.show();
    std::cout << "Sequential:" << std::endl;
    solution1.show();
}

