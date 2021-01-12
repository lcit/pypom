#!/bin/bash

cd pypom
mkdir build
rm build/*
cd build
cmake ..
make
cp core.so ..
cd ../..
