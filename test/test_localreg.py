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
along with localreg.  If not, see <http://www.gnu.org/licenses/>.
"""

from localreg import *
import numpy as np
import pytest

#
# PARAMETRIC KERNEL TESTS
#

all_kernels = [rectangular
              ,triangular
              ,epanechnikov
              ,biweight
              ,triweight
              ,tricube
              ,gaussian
              ,cosine
              ,logistic
              ,sigmoid
              ,silverman]

@pytest.mark.parametrize("kernel", [k for k in all_kernels if k!=silverman])
def test_nonnegative(kernel):
    t = np.linspace(-5, 5, 100)
    assert np.all(kernel(t)>=0)

@pytest.mark.parametrize("kernel", all_kernels)
def test_symmetry(kernel):
    t = np.linspace(-5, 5, 100)
    assert np.allclose(kernel(t), kernel(-t))

@pytest.mark.parametrize("kernel", all_kernels)
def test_normalized(kernel):
    t = np.linspace(-20, 20, 5000)
    dt = t[1]-t[0]
    int = np.sum(kernel(t))*dt
    assert int == pytest.approx(1, 1e-3)

#
# SIMPLE KERNEL TESTS
#

@pytest.fixture
def t():
    return np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

def test_rectangular(t):
    assert np.allclose(rectangular(t),
                       np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]))

def test_triangular(t):
    assert np.allclose(triangular(t),
                       np.array([0, 0, 0.5, 1, 0.5, 0, 0]))

def test_epanechnikov(t):
    assert np.allclose(epanechnikov(t),
                       0.75*np.array([0, 0, 0.75, 1, 0.75, 0, 0]))

def test_biweight(t):
    assert np.allclose(biweight(t),
                       (15/16)*np.array([0, 0, 0.75**2, 1, 0.75**2, 0, 0]))

def test_triweight(t):
    assert np.allclose(triweight(t),
                       (35/32)*np.array([0, 0, 0.75**3, 1, 0.75**3, 0, 0]))

def test_tricube(t):
    assert np.allclose(tricube(t),
                       (70/81)*np.array([0, 0, 0.875**3, 1, 0.875**3, 0, 0]))

def test_gaussian(t):
    assert np.allclose(gaussian(t),
                       (1/np.sqrt(2*np.pi))*np.array([np.exp(-1.125), np.exp(-0.5), np.exp(-0.125), 1, np.exp(-0.125), np.exp(-0.5), np.exp(-1.125)]))

def test_cosine(t):
    assert np.allclose(cosine(t),
                       (np.pi/4)*np.array([0, 0, np.sqrt(2)/2, 1, np.sqrt(2)/2, 0, 0]))

def test_logistic(t):
    assert np.allclose(logistic(t),
                       np.array([0.149146, 0.196612, 0.235004, 0.25, 0.235004, 0.196612, 0.149146]))

def test_sigmoid(t):
    assert np.allclose(sigmoid(t),
                       (2/np.pi)*np.array([0.212548, 0.324027, 0.443409, 0.5, 0.443409, 0.324027, 0.212548]))

def test_silverman(t):
    assert np.allclose(silverman(t),
                       0.5*np.array([0.333193, 0.491558, 0.637724, np.sqrt(2)/2, 0.637724, 0.491558, 0.333193]))
