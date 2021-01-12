#!/bin/bash

# change these...
PYTHON="/usr/local/Cellar/python@3.9/3.9.1_6/Frameworks/Python.framework/Versions/3.9"
PYTHON_LIBRARIES="${PYTHON}/lib/libpython3.9.dylib"
PYTHON_INCLUDE_DIRS="${PYTHON}/include/python3.9"

# for libboost_python3 and libboost_numpy3
Boost_LIB_python="/usr/local/Cellar/boost-python3/1.75.0/lib/libboost_python39.dylib"
Boost_LIB_numpy="/usr/local/Cellar/boost-python3/1.75.0/lib/libboost_numpy39.dylib"
Boost_LIBRARIES=${Boost_LIB_python}" "${Boost_LIB_numpy}
Boost_INCLUDE_DIRS="/usr/local/Cellar/boost/1.75.0_1/include"

cd pypom
mkdir build
rm build/*
cd build
cmake -DPYTHON_MANUAL=ON -DPYTHON_LIBRARIES=${PYTHON_LIBRARIES} -DPYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIRS} -DBoost_LIBRARIES=${Boost_LIBRARIES} -DBoost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}  ..
make
cp core.so ..
cd ../..
