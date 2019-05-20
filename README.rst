localreg
=========

.. image:: https://travis-ci.com/sigvaldm/localreg.svg?branch=master
    :target: https://travis-ci.com/sigvaldm/localreg

.. image:: https://coveralls.io/repos/github/sigvaldm/localreg/badge.svg?branch=master
    :target: https://coveralls.io/github/sigvaldm/localreg?branch=master

.. image:: https://img.shields.io/pypi/pyversions/localreg.svg
    :target: https://pypi.org/project/localreg

Smoothing of noisy data series through *local polynomial regression* (including LOESS/LOWESS).

Installation
------------
Install from PyPI using ``pip`` (preferred method)::

    pip install localreg

Or download the GitHub repository https://github.com/sigvaldm/localreg.git and run::

    python setup.py install

Introduction
------------
Local polynomial regression is performed using the function
``localreg(x, y, x0=None, degree=2, kernel=epanechnikov, width=1, frac=None)``, where
``x`` and ``y`` are the :math:``x`` and :math:``y``-values of the data to smooth, respectively. ``x0`` is the :math:``x``-values at which to compute the smoothed :math:``y``-values. By default this is the same as ``x``, but beware that the run time is proportional to the size of ``x0``, so if you have many datapoints, it may be worthwhile to specify a smaller ``x0`` yourself. ``degree`` is the degree of the polynomial which is locally fitted to the data, e.g., ``degree=1`` means you get local linear regression. For ``degree=0`` the method reduces to a weighted moving average. ``kernel`` is the kernel function (or weighting window) and is a pure function. In how big a neighborhood the local regression is to be performed can be specified by either ``width`` or ``frac``. ``width`` scales the width of the weighting kernels, and for kernels with compact support, this is actually half of the width. For Gaussian windows, etc., it is the "standard deviation". If ``frac`` is specified, ``width`` is ignored, and is instead set to be big enough to use the specified fraction of all datapoints, e.g., ``frac=0.5`` means that half of the datapoints are used at any time.

Implemented weighting functions are:
    - ``rectangular``
    - ``triangular``
    - ``epanechnikov``
    - ``biweight``
    - ``triweight``
    - ``tricube``
    - ``gaussian``
    - ``cosine``
    - ``logistic``
    - ``sigmoid``
    - ``silverman``

Example Usage
-------------
::

    import numpy as np
    import matplotlib.pyplot as plt
    from localreg import *

    np.random.seed(1234)
    x = np.linspace(1.5, 5, 2000)
    yf = np.sin(x*x)
    y = yf + 0.5*np.random.randn(*x.shape)

    y0 = localreg(x, y, degree=0, kernel=tricube, width=0.3)
    y1 = localreg(x, y, degree=1, kernel=tricube, width=0.3)
    y2 = localreg(x, y, degree=2, kernel=tricube, width=0.3)

    plt.plot(x, y, '+', markersize=0.6, color='gray')
    plt.plot(x, yf, label='Ground truth ($\sin(x^2)$)')
    plt.plot(x, y0, label='Moving average')
    plt.plot(x, y1, label='Local linear regression')
    plt.plot(x, y2, label='Local quadratic regression')
    plt.legend()
    plt.show()

.. image:: examples/basic.png



.. [Hastie] T. Hastie, R. Tibshirani and J. Friedman *The Elements of Statistical Learing -- Data Mining, Inference, and Prediction*, Second Edition, Springer, 2017.
.. [Cleveland] W. Cleveland *Robust Locally Weighted Regression and Smoothing Scatterplots*, Journal of the Americal Statistical Associations, 74, 1979.
