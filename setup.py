#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from pipenv.project import Project
from pipenv.utils import convert_deps_to_pip
from setuptools import setup
import unittest

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README'), encoding='utf-8') as f:
    long_description = '\n' + f.read()
    
# the Pipfile acts as the requirments.txt
pfile = Project(chdir=False).parsed_pipfile
requirements = convert_deps_to_pip(pfile['packages'], r=False)
test_requirements = convert_deps_to_pip(pfile['dev-packages'], r=False)  

setup(
    name='pypom',
    version='1.0.0',
    description='Hanndy wrapper around C++ POM',
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
    test_suite='nose.collector',
    license='GPL'
)
