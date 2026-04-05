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
    
    // std::string file_name = argv[1];
    
    // Problem<Matrix, ValuesVector> problem1;
    // problem1.readMps(file_name);
    
    // SequentialDualSimplex solver1(problem1);
    // solver1.initDualSimplex();
    // solver1.presolve("minInfeas");
    // solver1.solve("elaborated");

    // std::cout << "Sequential:" << std::endl;
    // LPsolution solution1 = problem1.getSolution();
    // solution1.show();
    // std::string file_name = argv[1];
    

    // Problem<Matrix, ValuesVector> problem2;
    // problem2.readMps(file_name);
    
    // SequentialDualSimplex solver2(problem2);
    // solver2.initDualSimplex();
    // solver2.presolve("minInfeas");
    // solver2.solve("elaborated");

    // std::cout << "Sequential:" << std::endl;
    // LPsolution solution2 = problem2.getSolution();
    // solution2.show();

    
    // Problem<CudaSparseMatrix, CudaDataDenseVector> problem1;
    // problem1.readMps(file_name);
    
    // CudaDualSimplex solver1(problem1);
    // solver1.initDualSimplex();
    // solver1.presolve("minInfeas");
    // solver1.solve("elaborated");

    // std::cout << "Cuda:" << std::endl;
    // LPsolution solution1 = problem1.getSolution();
    // solution1.show();

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
    
    cusparseHandle_t sp_handle;

    cudssHandle_t cudss_handle;
    cudssConfig_t cudss_config;
    cudssData_t cudss_data;

    cudssHandle_t cudss_handle_T;
    cudssConfig_t cudss_config_T;
    cudssData_t cudss_data_T;

    cudaStream_t stream = NULL;
    cudaStreamCreate(&stream);

    cudssCreate(&cudss_handle);
    cudssSetStream(cudss_handle, stream);
    cudssDataCreate(cudss_handle, &cudss_data);
    cudssConfigCreate(&cudss_config);

    cudssCreate(&cudss_handle_T);
    cudssDataCreate(cudss_handle_T, &cudss_data_T);
    cudssConfigCreate(&cudss_config_T);

    cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
    cudssConfigSet(cudss_config_T, CUDSS_CONFIG_REORDERING_ALG,
                         &reorder_alg, sizeof(cudssAlgType_t));
    int ione = 1;
    cudssConfigSet(cudss_config_T, CUDSS_CONFIG_USE_MATCHING,
                         &ione, sizeof(int));
    cudssAlgType_t matching_alg = CUDSS_ALG_DEFAULT; // matching with scaling, same as CUDSS_ALG_5
    cudssConfigSet(cudss_config_T, CUDSS_CONFIG_MATCHING_ALG,
                         &matching_alg, sizeof(cudssAlgType_t));

    cusparseCreate(&sp_handle);

    CudaSparseMatrix A_cu(A);
    A_cu.genCsc(sp_handle);
    A_cu.createDescr();
    CudaDataDenseVector rhs_cu1(rhs1), rhs_cu2(rhs2);
    CudaDataDenseVector sol_cu1(m), sol_cu2(m), sol_cu3(m), sol_cu4(n);
    CudaIndexVector indexes(IndexVector({0, 1, 3}));

    
    A_cu.show();
    rhs_cu1.show();
    rhs_cu2.show();
    
    std::cout << "FULL_UPDATE" << std::endl;
    A.dotUpdate(
        rhs1, rhs1, 
        sol1, -1, 1, 
        IndexVector({0, 1, 3}), 
        SpmvOptions::UPDATE,
        false
    );
    A_cu.dotUpdate(
        sp_handle,
        rhs_cu1, rhs_cu1, 
        sol_cu1, -1, 1, 
        indexes, 
        SpmvOptions::UPDATE,
        false
    );
    sol1.show();
    sol_cu1.updateHostMem();
    sol_cu1.show();
    
    std::cout << "SET_UPDATE" << std::endl;
    A.dotUpdate(
        rhs2, rhs1, 
        sol2, -1, 1, 
        IndexVector({0, 1, 3}), 
        SpmvOptions::UPDATE,
        true
    );
    A_cu.dotUpdate(
        sp_handle,
        rhs_cu2, rhs_cu1, 
        sol_cu2, -1, 1, 
        indexes, 
        SpmvOptions::UPDATE,
        true
    );
    sol2.show();
    sol_cu2.updateHostMem();
    sol_cu2.show();

    std::cout << "FULL_UPDATE_T" << std::endl;
    A.dotUpdate(
        rhs1, rhs2, 
        sol3, 1, 0, 
        IndexVector({0, 1, 3}), 
        SpmvOptions::UPDATE_T,
        false
    );
    A_cu.dotUpdate(
        sp_handle,
        rhs_cu1, rhs_cu2, 
        sol_cu3, 1, 0, 
        indexes, 
        SpmvOptions::UPDATE_T,
        false
    );
    sol3.show();
    sol_cu3.updateHostMem();
    sol_cu3.show();

    std::cout << "SET_UPDATE_T" << std::endl;
    A.dotUpdate(
        rhs1, rhs2, 
        sol4, -1, 1, 
        IndexVector({0, 1, 3}), 
        SpmvOptions::UPDATE_T,
        true
    );
    A_cu.dotUpdate(
        sp_handle,
        rhs_cu1, rhs_cu2, 
        sol_cu4, -1, 1, 
        indexes, 
        SpmvOptions::UPDATE_T,
        true
    );
    sol4.show();
    sol_cu4.updateHostMem();
    sol_cu4.show();
    

    cudssDataDestroy(cudss_handle, cudss_data);
    cudssConfigDestroy(cudss_config);
    cudssDestroy(cudss_handle);

    cudssDataDestroy(cudss_handle_T, cudss_data_T);
    cudssConfigDestroy(cudss_config_T);
    cudssDestroy(cudss_handle_T);

    cudaStreamSynchronize(stream);
}

