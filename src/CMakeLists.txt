if(WITH_CUDA)
    target_sources(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/parser.hpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/sequentialDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sequentialDualSimplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/parallelDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/parallelDualSimplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/baseDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/baseDualSimplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/linalg.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cudaKernels.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cudaKernels.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LPsolution.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LPsolution.cpp
    )

    add_library(app_test STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/parser.hpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/sequentialDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sequentialDualSimplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/baseDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/baseDualSimplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cudaKernels.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cudaKernels.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/linalg.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cu
    )
    
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cudaKernels.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/cudaKernels.hpp
    PROPERTIES LANGUAGE CUDA)
    
else()
    target_sources(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/parser.hpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/sequentialDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sequentialDualSimplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/parallelDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/parallelDualSimplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/baseDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/baseDualSimplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linalg.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LPsolution.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LPsolution.cpp
    )

    add_library(app_test STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/parser.hpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/sequentialDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sequentialDualSimplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/baseDualSimplex.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/baseDualSimplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/problem.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linalg.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
    )
endif()


