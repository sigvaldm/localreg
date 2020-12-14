#!/usr/bin/env python
"""
Copyright 2019 Sigvald Marholm <marholm@marebakken.com>

This file is part of localreg.

localreg is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

localreg is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with localreg  If not, see <http://www.gnu.org/licenses/>.
"""

from setuptools import setup
from io import open # Necessary for Python 2.7

with open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

with open('.version') as f:
    version = f.read().strip()

setup(name='localreg',
      version=version,
      description='Local Polynomial Regression',
      long_description=long_description,
      author='Sigvald Marholm',
      author_email='marholm@marebakken.com',
      url='https://github.com/sigvaldm/localreg.git',
      packages=['localreg'],
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'dill',
                        'scikit-learn'],
      license='LGPL',
      classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        ]
     )

