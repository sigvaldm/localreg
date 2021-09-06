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

from localreg import localreg
from localreg.rbf import *
from localreg.localreg import polyfit
import numpy as np
import pytest

#
# MAIN TESTS
#

def test_polyfit_simple_linear():
    x = np.array([0,  0,  1, 1, 2, 2])
    y = np.array([-1, 1, -1, 3, -1, 5])
    x0 = np.array([0, 1, 2])
    assert np.allclose(polyfit(x, y, x0, degree=1), x0)

def test_polyfit_simple_average():
    x = np.array([0,  0,  1, 1, 2, 2])
    y = np.array([-1, 1, -1, 3, -1, 5])
    x0 = np.array([0, 1, 2])
    assert np.allclose(polyfit(x, y, x0, degree=0), [1, 1, 1])

def test_polyfit_exact_polynomial():
    x = np.random.rand(50, 2)
    y = x[:,0]*x[:,1]
    assert not np.allclose(polyfit(x, y, x, degree=1), y)
    assert np.allclose(polyfit(x, y, x, degree=2), y)

def test_polyfit_realistic():
    x = np.array([-6.89438   ,  7.94300378, 5.5221823 ,   9.77749217, -0.35979986,
                   2.01456739,  4.80691814, 3.22260756,  -7.12156073, -8.69959441])
    y = np.array([-1.74962299, -8.55733072, 8.56537608,   1.79095858,  4.43380336,
                 -14.63365203,  5.41264117, 9.69660297, -13.85424098,  0.42264531])
    x0 = np.array([1., 2., 3.])
    assert np.allclose(polyfit(x, y, x0, degree=0),
                       [-0.84728193, -0.84728193, -0.84728193], rtol=1e-3)
    assert np.allclose(polyfit(x, y, x0, degree=1),
                       [-0.85479608, -0.49940979, -0.14402349], rtol=1e-3)
    assert np.allclose(polyfit(x, y, x0, degree=2),
                       [0.65209843, 0.89184061, 1.05446368], rtol=1e-3)

def test_localreg_exact_polynomial():
    x = np.random.rand(50, 2)
    y = x[:,0]*x[:,1]
    assert not np.allclose(localreg(x, y, x, degree=1), y)
    assert np.allclose(localreg(x, y, x, degree=2), y)

def test_localreg_realistic():
    x = np.array([-6.89438   ,  7.94300378, 5.5221823 ,   9.77749217, -0.35979986,
                   2.01456739,  4.80691814, 3.22260756,  -7.12156073, -8.69959441])
    y = np.array([-1.74962299, -8.55733072, 8.56537608,   1.79095858,  4.43380336,
                 -14.63365203,  5.41264117, 9.69660297, -13.85424098,  0.42264531])
    x0 = np.array([2., 3.])

    # Testing all orders
    assert np.allclose(localreg(x, y, x0, degree=0, kernel=epanechnikov, radius=1),
                       [-14.63365203, 8.9780852], rtol=1e-3)
    assert np.allclose(localreg(x, y, x0, degree=1, kernel=epanechnikov, radius=1),
                       [-14.5487543 , 5.21322664], rtol=1e-3)
    assert np.allclose(localreg(x, y, x0, degree=2, kernel=epanechnikov, radius=1),
                       [-14.4523815 , 3.77134959], rtol=1e-3)

    # Testing radius
    assert np.allclose(localreg(x, y, x0, degree=2, kernel=epanechnikov, radius=2),
                       [ -14.80997735,   7.00785276], rtol=1e-3)

    # Testing frac
    assert np.allclose(localreg(x, y, x0, degree=2, kernel=epanechnikov, frac=0.5),
                       [-6.21823369,  4.33953829], rtol=1e-3)

def test_localreg_narrow_kernel(caplog):
    x = np.array([0., 1., 2.])
    y = np.array([0., 1., 2.])
    x0 = np.array([0.5])
    y0 = localreg(x, y, x0, degree=2, kernel=epanechnikov, radius=0.4)
    assert np.isnan(y0)[0]
    assert(len(caplog.records) == 1)

def test_localreg_integer():
    x = np.array([0, 3, 6, 9], dtype=int)
    y = 0.5*x # Simple linear function should be exactly matched by degree=1
    x0 = np.array([1], dtype=int)
    y0 = localreg(x, y, x0, degree=1, radius=3)
    assert y0[0]==pytest.approx(0.5)
    # assert np.allclose(x0, y0)

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
