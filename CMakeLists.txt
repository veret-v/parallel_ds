cmake_minimum_required (VERSION 3.20)
project (solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmakeFindModules")

option(BUILD_TESTS "Build tests" ON)
option(WITH_CUDA "Build with CUDA support" ON)

# testing
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# COIN Utils library
find_package(CoinUtils REQUIRED)
include_directories(${COINUTILS_INCLUDE_DIRS})

add_executable(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} ${COINUTILS_LIBRARY})

# OpenMP
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP support found")
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()

# CUDA
if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_compile_definitions(WITH_CUDA)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_CUDA)

    target_link_libraries(${PROJECT_NAME} CUDA::cudart rt pthread dl)
    
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_ARCHITECTURES "native" 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

target_sources(${PROJECT_NAME} PRIVATE "src/main.cpp" )
target_include_directories(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}")

add_subdirectory("src")

if(BUILD_TESTS)

    add_executable(tests "tests/simple_tests_DS.cpp")
    target_link_libraries(tests PRIVATE app_test gtest_main)
    target_include_directories(tests PRIVATE "${PROJECT_SOURCE_DIR}")

    include(GoogleTest)
    gtest_discover_tests(tests)
    
endif(BUILD_TESTS)






