if(WITH_CUDA)
    target_sources(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaDualSimplex.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaSparseMatrix.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaSparseMatrix.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaDenseVector.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaDenseVector.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaKernels.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaKernels.hpp
    )
    
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaDualSimplex.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaSparseMatrix.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaSparseMatrix.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaDenseVector.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaDenseVector.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaKernels.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_version/cudaKernels.hpp
    PROPERTIES LANGUAGE CUDA)
    
endif()

target_sources(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/parser.hpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/sequentialDualSimplex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/sequentialDualSimplex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/parallelDualSimplex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/parallelDualSimplex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/baseDualSimplex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/baseDualSimplex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/problem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/problem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/matrix.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/valuesVector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/valuesVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/linalg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/linalg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LPsolution.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LPsolution.cpp
)

add_library(app_test STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/parser.hpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/sequentialDualSimplex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/sequentialDualSimplex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/baseDualSimplex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/baseDualSimplex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/problem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/problem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/matrix.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/valuesVector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/valuesVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/linalg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_version/linalg.cpp
)

