# Python version of Probabilistic Occupancy Map (POM)

F. Fleuret, J. Berclaz, R. Lengagne and P. Fua, Multi-Camera People Tracking with a Probabilistic Occupancy Map, IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 30, Nr. 2, pp. 267 - 282, February 2008.

https://www.epfl.ch/labs/cvlab/software/tracking-and-modelling-people/pom/

## Prerequisites

- numpy
- scipy
- imageio
- pyyaml
- matplotlib
- python 3.5+
- CMake (c++ only if built manually)
- Boost.Python 1.66+

## Installation

* Follow the instruction on how to install Boost.Python in the section below
* Compile the core using `build.sh`
  * In case of error try to compile the core manually. Have a look at `build_manually_cmake.sh` or `build_manually_macos.sh`. You will have to find the location of the headers and libraries of python, boost-python and boost-numpy and modify paths in these files. Try `sudo find / -name "libboost_python*"` if don't know where to look.
* If the core is compiled correctly you should be able to run this `cd pypom; python -c "import core; help(core)"`
* Add the package in your environmental variable `export PYTHONPATH="~/code/pypom:$PYTHONPATH"`
* That's it! You should be able to import pypom in python.

### Installing Boost.Python

#### Linux

We strongly suggest installing Boost.Python from source as the headers files are required to compile our code:
```
cd $HOME
wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.bz2
tar --bzip2 -xf boost_1_66_0.tar.bz2
cd boost_1_66_0
./bootstrap.sh --with-python=python3
sudo ./b2 install
sudo ldconfig
```

###### Possible issues:
1. If you have a message similar to this one: `failed updating 66 targets...fatal error: pyconfig.h: No such file or directory` then you need to modify line 542 of file tools/build/src/tools/python.jam from this:
		includes ?= $(prefix)/include/python$(version) ;
to this:
		includes ?= $(prefix)/include/python$(version)m ;
now try again installing boost.python:
```
./bootstrap.sh --with-python=python3
sudo ./b2 install
sudo ldconfig
```

1. Another version of Boost is clashing with the new one. If it is the case for you we suggest removing the old version and re-install the new one. HINT: after compiling the core module, type `ldd core.so` to see which boost.python was used.

#### MacOS 

On MacOS Boost.Python can be installed with Homebrew:
```
brew install boost
brew install boost-python3
brew install cmake
```
take note of which python version it has been used to install boost-python! If you have multiple python installed or Anaconda you have to make sure, while compiling this project, that the correct versions are used.
If there is a mismatch of version try to build this project manaully. Have a look at the files `build_manually_cmake.sh` or `build_manually_macos.sh`.

#### How to find Boost and Boost.Python
If you have no clue where to look:
```
sudo find / -name "libboost_python*"
sudo find / -name "libboost_numpy*"
sudo find / -name "libboost*"
```

## License

GPL v3

### Things to know and advices

* placing the cameras at an altitude is better than close to the floor
* the more the cameras the more the people you can detect correctly
* place the cameras so as the picture is straight otherwise the rectangles will be larger and performance will degrade
* obstacles such as walls or forniture are not handeled by the algorithm. This means that it will fail in such a scenarios
* using a CNN to perform background subtraction is in general the best option
* shadows are really bad. Try to avoid them as much as possible
* make sure the cameras are well calibrated and synchronized


## Other

### Installing CMake with SSL support
The installation of CMake with SSL support is required for the installation of OpenCV with the special Intel IPP functions.
```
sudo apt-get install libcurl4-gnutls-dev
mdkir -p ~/tmp/cmake
cd !$
wget https://cmake.org/files/v3.10/cmake-3.10.2.tar.gz
tar -xzf cmake-3.10.2.tar.gz
cd cmake-3.10.2/
./bootstrap --parallel=8 --system-curl
make -j 8
```
then we remove the old CMake with:
```
sudo apt remove cmake
```
and finally we install the new CMake with:
```
sudo make install
```