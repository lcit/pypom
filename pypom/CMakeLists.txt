cmake_minimum_required(VERSION 2.9)

set(PY_LIB_NAME "core")

set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -O2 -Wp,-w")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
#find_package(NumPy REQUIRED)
include("${CMAKE_CURRENT_SOURCE_DIR}/FindNumpy.cmake")
find_package(Boost COMPONENTS python3)

include_directories(. .. ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

add_library(${PY_LIB_NAME} SHARED wrapper.cpp)
set_target_properties(${PY_LIB_NAME} PROPERTIES PREFIX "") # this removes the prefix lib 
if(WIN32)
  # Ensure the output binary is called example.pyd on Windows (not example.dll)
  set_target_properties(${PY_LIB_NAME} PROPERTIES SUFFIX ".pyd")
endif(WIN32)
target_link_libraries(${PY_LIB_NAME} boost_python3 boost_numpy3 ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})


