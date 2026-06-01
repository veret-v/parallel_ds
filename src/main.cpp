#include <filesystem>
#include <algorithm>

#include <omp.h>
#include <mpi.h>

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
    int solver_type = std::stoi(argv[2]);
    if (solver_type == 0)
    { 
        Problem<Matrix, ValuesVector> problem;
        problem.readMps(file_name);
        problem.transformToComputeForm();
        
        SequentialDualSimplex solver_seq(problem);
        solver_seq.initDualSimplex();
        solver_seq.presolve("minInfeas");
        solver_seq.solve();

        std::cout << "Sequential:" << std::endl;
        LPsolution solution_seq = problem.getSolution();
        solution_seq.show();
    } 
    else if (solver_type == 1)
    {
        Problem<CudaSparseMatrix, CudaDataDenseVector> problem_cuda;
        problem_cuda.readMps(file_name);
        problem_cuda.transformToComputeForm();
        
        CudaDualSimplex solver_cuda(problem_cuda);
        solver_cuda.initDualSimplex();
        solver_cuda.presolve("minInfeas");
        solver_cuda.solve();

        std::cout << "Cuda:" << std::endl;
        LPsolution solution_cuda = problem_cuda.getSolution();
        solution_cuda.show();
    }
    else if (solver_type == 2)
    {
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
        MPI_Comm_size(MPI_COMM_WORLD, &size);  

        Problem<Matrix, ValuesVector> problem;
        problem.readMps(file_name);
        problem.transformToComputeForm();
            
        if (rank == 0)
        {
            omp_set_num_threads(4);  

            ParallelDualSimplex solver_master(problem);
            solver_master.initMaster(0, size);
            solver_master.presolve("minInfeas");
            solver_master.solveMaster();

            std::cout << "MPI:" << std::endl;
            LPsolution solution = problem.getSolution();
            solution.show();
        }
        else
        {
            omp_set_num_threads(1);  

            ParallelDualSimplex solver_worker(problem);
            solver_worker.initWorker(0, rank);
            solver_worker.solveWorker();
        }
        
        MPI_Finalize();
    }
}

