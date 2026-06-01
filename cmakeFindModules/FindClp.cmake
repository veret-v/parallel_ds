find_path(CLP_INCLUDE_DIR
  NAMES ClpSimplex.hpp
  PATHS
    /usr/include/coin
    /usr/local/include/coin
    /opt/local/include/coin
    /usr/include
    /usr/local/include
  DOC "Clp include directory"
)

find_library(CLP_LIBRARY
  NAMES Clp
  PATHS
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /usr/lib/x86_64-linux-gnu
  DOC "Clp library"
)

if(CLP_INCLUDE_DIR AND CLP_LIBRARY)
  set(Clp_FOUND TRUE)
  set(CLP_INCLUDE_DIRS ${CLP_INCLUDE_DIR})
  set(CLP_LIBRARIES ${CLP_LIBRARY})
  message(STATUS "Found Clp: ${CLP_LIBRARY}")
else()
  set(Clp_FOUND FALSE)
  message(STATUS "Clp not found")
endif()