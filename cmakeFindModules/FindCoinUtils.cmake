find_path(COINUTILS_INCLUDE_DIR
  NAMES CoinPackedMatrix.hpp
  PATHS
    /usr/include/coin
    /usr/local/include/coin
    /opt/local/include/coin
    /usr/include
    /usr/local/include
  DOC "CoinUtils include directory"
)

find_library(COINUTILS_LIBRARY
  NAMES CoinUtils
  PATHS
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /usr/lib/x86_64-linux-gnu
  DOC "CoinUtils library"
)

if(COINUTILS_INCLUDE_DIR AND COINUTILS_LIBRARY)
  set(CoinUtils_FOUND TRUE)
  set(COINUTILS_INCLUDE_DIRS ${COINUTILS_INCLUDE_DIR})
  set(COINUTILS_LIBRARIES ${COINUTILS_LIBRARY})
  message(STATUS "Found CoinUtils: ${COINUTILS_LIBRARY}")
else()
  set(CoinUtils_FOUND FALSE)
  message(STATUS "CoinUtils not found")
endif()