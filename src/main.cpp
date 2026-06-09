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
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mps-file> <solver-type> [phi]" << std::endl;
        return 1;
    }
    std::string file_name = argv[1];
    int solver_type = std::stoi(argv[2]);
    double psi = 0.95;
    if (solver_type == 2) {
        if (argc < 4) {
            std::cerr << "Parallel solver requires phi parameter" << std::endl;
            return 1;
        }
        psi = std::stod(argv[3]);
    }

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
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        // Явно форсируем инициализацию контекста
        cudaFree(0);
        
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
            omp_set_num_threads(5);  

            ParallelDualSimplex solver_master(problem);
            solver_master.initMaster(0, size, psi);
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

