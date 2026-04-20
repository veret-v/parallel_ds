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
    
    Problem<Matrix, ValuesVector> problem2;
    problem2.readMps(file_name);
    
    SequentialDualSimplex solver2(problem2);
    solver2.initDualSimplex();
    solver2.presolve("minInfeas");
    solver2.solve("elaborated");

    std::cout << "Sequential:" << std::endl;
    LPsolution solution2 = problem2.getSolution();
    solution2.show();

    
    Problem<CudaSparseMatrix, CudaDataDenseVector> problem1;
    problem1.readMps(file_name);
    
    CudaDualSimplex solver1(problem1);
    solver1.initDualSimplex();
    solver1.presolve("minInfeas");
    solver1.solve("elaborated");

    std::cout << "Cuda:" << std::endl;
    LPsolution solution1 = problem1.getSolution();
    solution1.show();

    //  int m = 3, n =3;
    // ValuesVector rhs1(std::vector<double>{1.5, 2, 0.5});
    // std::vector<double> expected = {1.25, 0.5, -0.5};
    // ValuesVector sol1(n);
    // Matrix A(
    //     std::vector<double>{1, 0.5, 2, 1, 1},
    //     std::vector<MKL_INT>{0, 2, 4, 5},
    //     std::vector<MKL_INT>{0, 1, 0, 2, 1},
    //     m, n
    // );
    
    // cusparseHandle_t sp_handle;

    // cudssHandle_t cudss_handle;
    // cudssConfig_t cudss_config;
    // cudssData_t cudss_data;

    // cudssHandle_t cudss_handle_T;
    // cudssConfig_t cudss_config_T;
    // cudssData_t cudss_data_T;

    // cudaStream_t stream = NULL;
    // cudaStreamCreate(&stream);

    // cudssCreate(&cudss_handle);
    // cudssSetStream(cudss_handle, stream);
    // cudssDataCreate(cudss_handle, &cudss_data);
    // cudssConfigCreate(&cudss_config);

    // cudssCreate(&cudss_handle_T);
    // cudssDataCreate(cudss_handle_T, &cudss_data_T);
    // cudssConfigCreate(&cudss_config_T);

    // cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
    // cudssConfigSet(cudss_config_T, CUDSS_CONFIG_REORDERING_ALG,
    //                      &reorder_alg, sizeof(cudssAlgType_t));
    // int ione = 1;
    // cudssConfigSet(cudss_config_T, CUDSS_CONFIG_USE_MATCHING,
    //                      &ione, sizeof(int));
    // cudssAlgType_t matching_alg = CUDSS_ALG_DEFAULT; // matching with scaling, same as CUDSS_ALG_5
    // cudssConfigSet(cudss_config_T, CUDSS_CONFIG_MATCHING_ALG,
    //                      &matching_alg, sizeof(cudssAlgType_t));

    // cusparseCreate(&sp_handle);

    // CudaSparseMatrix A_cu(A);

    // A_cu.genCsc(sp_handle);
    // A_cu.createDescr();

    // CudaDataDenseVector rhs_cu(rhs1), sol_cu(n);

    // A_cu.LUdecompose(
    //     cudss_handle, cudss_config, cudss_data,
    //     cudss_handle_T, cudss_config_T, cudss_data_T
    // );
    // A_cu.solve(
    //     cudss_handle_T, cudss_config_T, cudss_data_T,
    //     rhs_cu, sol_cu, true
    // );
    // sol_cu.updateHostMem();

    // A.show();
    // rhs_cu.show();

    // A.LUdecompose();
    // A.solve(rhs1, sol1, true);

    // sol1.show();
    // sol_cu.show();

    // cudssDataDestroy(cudss_handle, cudss_data);
    // cudssConfigDestroy(cudss_config);
    // cudssDestroy(cudss_handle);

    // cudssDataDestroy(cudss_handle_T, cudss_data_T);
    // cudssConfigDestroy(cudss_config_T);
    // cudssDestroy(cudss_handle_T);

    // cudaStreamSynchronize(stream);


}

