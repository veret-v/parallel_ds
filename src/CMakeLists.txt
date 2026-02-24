if(WITH_CUDA)
    target_sources(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaDualSimplex.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaSparseMatrix.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaSparseMatrix.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaDenseVector.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaDenseVector.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaIndexVector.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaIndexVector.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaDataDenseVector.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaDataDenseVector.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaKernels.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaKernels.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/PFIfactor.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/PFIfactor.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utillities.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/utillities.hpp
    )
    
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaDualSimplex.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaSparseMatrix.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaSparseMatrix.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaDenseVector.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaDenseVector.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaKernels.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/cudaKernels.hpp
    PROPERTIES LANGUAGE CUDA)
    
endif()

target_sources(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/sequentialDualSimplex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/sequentialDualSimplex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/parallelDualSimplex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/parallelDualSimplex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/baseDualSimplex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/baseDualSimplex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/problem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/problem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/matrix.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/valuesVector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/valuesVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/linalg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/linalg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/LPsolution.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/LPsolution.cpp
)

add_library(app_test STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/sequentialDualSimplex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/sequentialDualSimplex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/baseDualSimplex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/baseDualSimplex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/problem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/problem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/matrix.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/valuesVector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/valuesVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/linalg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/linalg.cpp
)

