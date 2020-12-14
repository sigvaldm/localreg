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

from localreg.rbf import *
from localreg.metrics import *
from localreg import *
import numpy as np
import matplotlib.pyplot as plt
import pytest
from pytest import approx

def test_simple():
    x = [1,2]
    y = [1,1]
    net = RBFnet()
    net.train(x, y, radius=1, num=2, normalize=False, rbf=triangular)
    assert net.predict([1.5])==1

def test_complex():
    x = np.linspace(0,2*np.pi,100)
    y = np.sin(x) + 2
    net = RBFnet()
    net.train(x[:,None], y)
    y_hat = net.predict(x[:,None])
    assert max_rel_error(y, y_hat) < 1e-6

def test_eval_bases():
    x = np.array([1,2]).reshape(-1,1)
    y = np.array([1,1]).reshape(-1,1)
    net = RBFnet()
    net.train(x, y, radius=1, num=2, normalize=False, rbf=triangular)
    bases = net.eval_bases(np.array([1.5]).reshape(-1,1))
    assert np.allclose(bases, [0.5, 0.5])

def test_plot_corr():
    x = np.array([0, 1, 2])
    y = np.array([-0.1, 1.1, 1.9])
    fig, ax = plt.subplots()
    plot_corr(ax, x, y)
    x_, y_ = ax.lines[0].get_xydata().T
    assert np.array_equal(x_, x)
    assert np.array_equal(y_, y)
    x_, y_ = ax.lines[1].get_xydata().T
    assert np.array_equal(x_, [-0.1, 2])
    assert np.array_equal(x_, y_)

def test_plot_centers():
    net = RBFnet()
    net.centers = np.array([[1,10], [2,20]])
    fig, ax = plt.subplots()
    net.plot_centers(ax)
    x_, y_ = ax.lines[0].get_xydata().T
    assert np.array_equal(x_, [1,2])
    assert np.array_equal(y_, [10,20])

def test_plot_bases():
    x = np.array([1,2]).reshape(-1,1)
    y = np.array([1,1]).reshape(-1,1)
    net = RBFnet()
    net.train(x, y, radius=1, num=2, normalize=False, rbf=triangular)

    x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
    fig, ax = plt.subplots()
    net.plot_bases(ax, x.reshape(-1,1))

    x0, y0 = ax.lines[0].get_xydata().T # Full line
    x1, y1 = ax.lines[1].get_xydata().T # Basis function 1
    x2, y2 = ax.lines[2].get_xydata().T # Basis function 2

    assert np.array_equal(x0, x)
    assert np.array_equal(x1, x)
    assert np.array_equal(x2, x)
    assert np.allclose(y0, [0, 0.5, 1, 1, 1, 0.5, 0])

    # The basis functions should look like this, but which is which is unknown
    ya = [0, 0.5, 1, 0.5, 0, 0, 0]
    yb = [0, 0, 0, 0.5, 1, 0.5, 0]

    assert (np.allclose(y1, ya) and np.allclose(y2, yb)) or \
           (np.allclose(y1, yb) and np.allclose(y2, ya))

def test_error_metrics():
    true = np.array([1.,1,2])
    pred = np.array([1.,-1,3])
    assert rms_error(true, pred) == approx(np.sqrt(5/3))
    assert rms_rel_error(true, pred) == approx(np.sqrt(17./12))
    assert max_abs_error(true, pred) == approx(2)
    assert mean_abs_error(true, pred) == approx(1)
    assert max_rel_error(true, pred) == approx(2)
    assert mean_rel_error(true, pred) == approx(5./6)
    assert error_bias(true, pred) == approx(-1./3)
    assert rel_error_bias(true, pred) == approx(-0.5)
    assert error_std(true, pred) == approx(np.sqrt(14)/3)
    assert rel_error_std(true, pred) == approx(np.sqrt(14./12))

def test_save_load():
    net = RBFnet()
    x = np.array([1,2]).reshape(-1,1)
    y = np.array([1,1]).reshape(-1,1)
    net = RBFnet()
    net.train(x, y, radius=1, num=2, normalize=False, rbf=triangular)
    net.save('tmp.npz')
    net = RBFnet()

    # Check that no data loaded to RBFnet
    with pytest.raises(AttributeError):
        assert net.predict(np.array([1.5]).reshape(-1,1))==1

    net.load('tmp.npz')
    assert net.predict(np.array([1.5]).reshape(-1,1))==1

def test_relative_least_squares():
    x = np.linspace(0,0.49,100)
    y = np.tan(np.pi*x)+1
    net = RBFnet()

    net.train(x[:,None], y, radius=1)
    y_hat = net.predict(x[:,None])
    err = rms_error(y, y_hat)
    err_rel = rms_rel_error(y, y_hat)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(x, y)
    # plt.plot(x, y_hat)

    net.train(x[:,None], y, radius=1, relative=True)
    y_hat = net.predict(x[:,None])
    err_r = rms_error(y, y_hat)
    err_rel_r = rms_rel_error(y, y_hat)

    # plt.plot(x, y_hat)
    # plt.show()

    assert err < err_r*0.9
    assert err_rel_r < err_rel*0.9

def test_keep_aspect():
    net = RBFnet()
    input = np.array([[-1, 0],
                      [ 1, 0],
                      [ 0, 2],
                      [ 0,-2],
                      [ 0, 0]]) + 1
    output = np.array([0, 0, 0, 0, 0])

    net.adapt_normalization(input, output)
    assert np.allclose(net.input_scale, np.sqrt([0.4, 1.6]))

    net.adapt_normalization(input, output, keep_aspect=True)
    assert net.input_scale == approx(np.sqrt(2))
