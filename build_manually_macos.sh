#!/bin/bash

# change these..
PYTHON_MAJOR_VERSION=3
PYTHON_MINOR_VERSION=9
PYTHON="/usr/local/Cellar/python@3.9/3.9.1_6/Frameworks/Python.framework/Versions/3.9"
LIB_PYTHON=${PYTHON}"/lib"
INCLUDE_PYTHON=${PYTHON}"/include/python3.9"
INCLUDE_BOOST="/usr/local/Cellar/boost/1.75.0_1/include"

cd pypom
c++ -std=c++14 -Wp,-w -shared wrapper.cpp -L${LIB_PYTHON} -lpython${PYTHON_MAJOR_VERSION}.${PYTHON_MINOR_VERSION} -lboost_python${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION} -lboost_numpy${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION} -o core.so -I. -I${INCLUDE_PYTHON} -I${INCLUDE_BOOST}
cd ..
