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
Local polynomial regression is performed using the function::

    localreg(x, y, x0=None, degree=2, kernel=epanechnikov, width=1, frac=None)

where ``x`` and ``y`` are the x and y-values of the data to smooth, respectively.
``x0`` is the x-values at which to compute smoothed values. By default this is the same as ``x``, but beware that the run time is proportional to the size of ``x0``, so if you have many datapoints, it may be worthwhile to specify a smaller ``x0`` yourself.

Local polynomial regression works by fitting a polynomial of degree ``degree`` to the datapoints in vicinity of where you wish to compute a smoothed value (``x0``), and then evaluating that polynomial at ``x0``. For ``degree=0`` it reduces to a weighted moving average. A weighting function or kernel ``kernel`` is used to assign a higher weight to datapoints near ``x0``. The argument to ``kernel`` is a pure function of one argument so it is possible to define custom kernels. The following kernels are already implemented:

    - ``rectangular``
    - ``triangular``
    - ``epanechnikov``
    - ``biweight``
    - ``triweight``
    - ``tricube``
    - ``gaussian`` (non-compact)
    - ``cosine``
    - ``logistic`` (non-compact)
    - ``sigmoid`` (non-compact)
    - ``silverman`` (non-compact)

Having a kernel wich tapers off toward the edges, i.e., not a rectangular kernel, results in a smooth output.

The width of the kernel can be scaled by the parameter ``width``, which is actually half of the kernel-width for kernels with compact support. For kernels with non-compact support, like the Gaussian kernel, it is simply a scaling parameter, akin to the standard deviation. Having a wider kernel and including more datapoints lowers the noise (variance) but increases the bias as the regression will not be able to capture variations on a scale much narrower than the kernel window.

For unevenly spaced datapoints, having a fixed width means that a variable number of datapoints are included in the window, and hence the noise/variance is variable too. However, the bias is fixed. Using a width that varies such that a fixed number of datapoints is included leads instead to constant noise/variance but fixed bias. This can be acheived by specifying ``frac`` which overrules ``width`` and specifies the fraction of all datapoints to be included in the width of the kernel.

Example Usage
-------------
The below example exhibits several interesting features::

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

If there's a slope in the data near an edge, a simple moving average will fail to take into account the slope, as seen in the figure, since most of the datapoints will be to the right (or left) of ``x0``. A local linear (or higher order regression) is able to compensate for this. We also see that as the frequency of the oscillations increases, the local linear regression is not able to keep up, because the variations become too small compared to the window. A smaller window would help, at the cost of more noise in the regression. Another option is to increase the degree to 2. The quadratic regression is better at filling the valleys and the hills. For too rapid changes compared to the kernel, however, quadratic polynomials will also start failing.

It is also worth noting that a higher degree also comes with an increase in variance, which can show up as small spurious oscillations. It is therefore not very common to go higher than 2, although localreg supports arbitrary degree.

.. [Hastie] T. Hastie, R. Tibshirani and J. Friedman *The Elements of Statistical Learing -- Data Mining, Inference, and Prediction*, Second Edition, Springer, 2017.
.. [Cleveland] W. Cleveland *Robust Locally Weighted Regression and Smoothing Scatterplots*, Journal of the Americal Statistical Associations, 74, 1979.
