target_sources(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/parser.hpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/sequentialDualSimplex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sequentialDualSimplex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/problem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/problem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linalg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)


add_library(app_test STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/parser.hpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/sequentialDualSimplex.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sequentialDualSimplex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/problem.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/problem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/valuesVector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linalg.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
)
