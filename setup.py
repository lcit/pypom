#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from pipenv.project import Project
from pipenv.utils import convert_deps_to_pip
from setuptools import setup, Extension
import unittest
from distutils.command.build_ext import build_ext as build_ext_orig
import pathlib

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README'), encoding='utf-8') as f:
    long_description = '\n' + f.read()
    
# the Pipfile acts as the requirments.txt
pfile = Project(chdir=False).parsed_pipfile
requirements = convert_deps_to_pip(pfile['packages'], r=False)
test_requirements = convert_deps_to_pip(pfile['dev-packages'], r=False)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class build_ext(build_ext_orig):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        # current working directory
        cwd = pathlib.Path().absolute()

        # the folder where to build pom core
        build_folder = pathlib.Path("pypom/build")
        build_folder.mkdir(parents=True, exist_ok=True)

        # move to build folder
        os.chdir(str(build_folder))

        # build cmake
        self.spawn(['cmake', '..'])

        # build lib
        self.spawn(['make'])

        # copy the lib into the parent folder
        self.spawn(['cp', 'core.so', '..'])

        # come back
        os.chdir(str(cwd)) 

setup(
    name='pypom',
    version='1.0.0',
    description='Python version of Probabilistic Occupancy Map',
    long_description=long_description,
    author='Leonardo Citraro',
    author_email='leonardo.citraro@epfl.ch',
    url='',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.5'
    ],
    packages=['pypom'],
    install_requires=requirements,
    tests_require=test_requirements,
    include_package_data=True,
    ext_modules=[CMakeExtension('cmake_pom_core')],
    cmdclass={
        'build_ext': build_ext,
    },
    test_suite='nose.collector',
    license='GPL'
)

