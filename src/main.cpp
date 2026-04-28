#include <filesystem>
#include <algorithm>

#include "./cpu/valuesVector.hpp"
#include "./cpu/matrix.hpp"
#include "./common/problem.hpp"
#include "./cpu/sequentialDualSimplex.hpp"
#include "./cpu/parallelDualSimplex.hpp"

#include "./gpu/cudaDataDenseVector.hpp"
#include "./gpu/cudaSparseMatrix.hpp"
#include "./gpu/cudaDualSimplex.hpp"


namespace fs = std::filesystem;

class Runner
{
private:
    /* data */
public:
    Runner(/* args */);
    ~Runner();
};

Runner::Runner(/* args */)
{
}

Runner::~Runner()
{
}


int main(int argc, char* argv[])
{
    
    std::string file_name = argv[1];
    int run_cuda = std::stoi(argv[2]);
    if (!run_cuda)
    { 
        Problem<Matrix, ValuesVector> problem2;
        problem2.readMps(file_name);
        
        SequentialDualSimplex solver2(problem2);
        solver2.initDualSimplex();
        solver2.presolve("minInfeas");
        solver2.solve("elaborated");

        std::cout << "Sequential:" << std::endl;
        LPsolution solution2 = problem2.getSolution();
        solution2.show();
    } 
    else
    {
        Problem<CudaSparseMatrix, CudaDataDenseVector> problem1;
        problem1.readMps(file_name);
        
        CudaDualSimplex solver1(problem1);
        solver1.initDualSimplex();
        solver1.presolve("minInfeas");
        solver1.solve("elaborated");

        std::cout << "Cuda:" << std::endl;
        LPsolution solution1 = problem1.getSolution();
        solution1.show();
    }
}

