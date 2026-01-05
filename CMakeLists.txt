cmake_minimum_required (VERSION 3.20)
project (solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmakeFindModules")

find_package(CoinUtils REQUIRED)

include_directories(${COINUTILS_INCLUDE_DIRS})

add_executable(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} ${COINUTILS_LIBRARY})

target_sources(${PROJECT_NAME} PRIVATE "src/main.cpp" )
target_include_directories(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}")
add_subdirectory("src")
