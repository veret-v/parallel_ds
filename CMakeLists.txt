cmake_minimum_required (VERSION 3.20)
project (solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmakeFindModules")

option(BUILD_TESTS "Build tests" OFF)
option(WITH_CUDA "Build with CUDA support" ON)
option(WITH_MPI "Build with MPI support" ON)
option(WITH_OPENMP "Build with OPENMP support" ON)
option(WITH_AVX "Build with AVX support" ON)

add_executable(${PROJECT_NAME})

# COIN Utils library
find_package(CoinUtils REQUIRED)
include_directories(${COINUTILS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${COINUTILS_LIBRARY})

find_package(Clp REQUIRED)
include_directories(${CLP_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CLP_LIBRARY})


# OpenMP
if(WITH_OPENMP)
    add_compile_definitions(WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

# MPI
if(WITH_MPI)
    add_compile_definitions(WITH_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(${PROJECT_NAME} MPI::MPI_CXX)
endif()

# Intel MKL
find_package(MKL REQUIRED)
if (MKL_FOUND)
    include_directories(${MKL_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} ${MKL_LIBRARIES})
endif()

# CUDA
if(WITH_CUDA)
    set(cudss_DIR "/usr/lib/x86_64-linux-gnu/cmake/cudss")

    find_package(CUDAToolkit REQUIRED)
    find_package(cudss REQUIRED)
   
    add_compile_definitions(WITH_CUDA)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_CUDA)

    target_link_libraries(${PROJECT_NAME} 
        CUDA::cudart 
        CUDA::cusolver
        CUDA::cublas 
        pthread dl)
    target_link_libraries(${PROJECT_NAME} /usr/lib/x86_64-linux-gnu/libcudss/12/libcudss.so.0.7.1)
    
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_ARCHITECTURES "native" 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

target_sources(${PROJECT_NAME} PRIVATE "src/main.cpp" )
target_include_directories(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}")

add_subdirectory("src")


# testing
if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(tests "tests/simple_tests_DS.cpp")
    target_link_libraries(tests PRIVATE app_test gtest_main)
    target_include_directories(tests PRIVATE "${PROJECT_SOURCE_DIR}")
    

    include(GoogleTest)
    gtest_discover_tests(tests)
    
endif(BUILD_TESTS)






